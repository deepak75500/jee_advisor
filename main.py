"""
main.py — JEE College Advisor + StudyCord Community
====================================================
Production v7.0 — Cloud-native stack
  • TiDB Cloud  : all structured data (posts, groups, messages, chat, caches)
  • MEGA        : binary media files only (images, videos, documents)
  • Groq        : AI counselor + prompt-engineered college insights + AI tasks
  • Selenium    : Perplexity.ai scraper for live college enquiry data (cached)

NEW in v7.0
  • Category/Caste support (OPEN, OBC-NCL, SC, ST, EWS) — filters data where
    caste-specific rows exist; falls back to OPEN rows with a user-facing note
    when the loaded dataset has only OPEN entries.
  • Advanced Rank Predictor: percentile→rank, marks→percentile, branch-wise
    probability matrix, year-over-year trend simulation, seat availability map.
  • Enhanced AI college recommendation prompts with placement tables, branch
    comparison matrices, and caste-aware quota guidance.
  • /api/rank-predictor  — new endpoint
  • /api/category-info   — caste/quota explainer endpoint
  • /api/college-compare — side-by-side AI comparison of 2–4 colleges

Architecture
  ┌─────────────┐   HTTP/WS   ┌──────────────────┐
  │  Client     │ ──────────► │  FastAPI (main)  │
  └─────────────┘             └────────┬─────────┘
                                       │
              ┌──────────────┬─────────┴──────┬──────────────┐
              ▼              ▼                ▼              ▼
           TiDB           MEGA            Groq API     Selenium
        (database.py)  (mega_store)   (llm prompts)  (scraper)
"""

from __future__ import annotations

import os, json, re, math, uuid, random, string, asyncio, threading, time
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Set, Tuple
import tempfile

import pandas as pd
from fastapi import (
    FastAPI, Request, HTTPException,
    WebSocket, WebSocketDisconnect,
    UploadFile, File, Query,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from groq import Groq

# ── Internal modules ──────────────────────────────────────────────────────────
from database import (
    init_db, db_purge_expired_groups,
    db_create_post, db_get_posts, db_count_posts, db_delete_post,
    db_get_post, db_toggle_reaction, db_add_comment, db_get_comments,
    db_delete_comment,
    db_create_group, db_get_groups_public, db_get_my_groups, db_get_group,
    db_get_group_by_code, db_delete_group, db_is_member, db_add_member,
    db_remove_member, db_get_members, db_has_request, db_add_request,
    db_get_requests, db_remove_request,
    db_send_message, db_get_messages, db_delete_message,
    db_get_college_insights, db_set_college_insights,
    db_get_scrape,
)
from p_scraper import scrape_college_info
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
# ── MEGA ──────────────────────────────────────────────────────────────────────
try:
    from mega import Mega
    _MEGA_AVAILABLE = True
except ImportError:
    _MEGA_AVAILABLE = False
    print("[WARN] mega.py not installed — media uploads disabled.")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

MEGA_EMAIL    = os.getenv("MEGA_EMAIL")
MEGA_PASSWORD = os.getenv("MEGA_PASSWORD")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")

POSTS_PER_PAGE = 30
MAX_FILE_SIZE  = 50 * 1024 * 1024  # 50 MB

STAGING_DIR = Path(tempfile.gettempdir()) / "sc_media_staging"
STAGING_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/webm", "video/quicktime", "video/x-matroska"}
ALLOWED_DOC_TYPES   = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
ALLOWED_ALL_TYPES = ALLOWED_IMAGE_TYPES | ALLOWED_VIDEO_TYPES | ALLOWED_DOC_TYPES

_EXT_MIME_MAP: Dict[str, str] = {
    ".pdf": "application/pdf", ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp",
    ".mp4": "video/mp4", ".webm": "video/webm",
    ".mov": "video/quicktime", ".mkv": "video/x-matroska",
}

_executor = ThreadPoolExecutor(max_workers=8)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
try:
    from google import genai
    GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
    GOOGLE_GEMINI_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-3.5-flash")
    gemini_client = genai.Client(api_key=GOOGLE_GEMINI_API_KEY) if GOOGLE_GEMINI_API_KEY else None
except Exception:
    genai = None
    gemini_client = None
    GOOGLE_GEMINI_API_KEY = None
    GOOGLE_GEMINI_MODEL = None


def _resolve_content_type(declared_ct: str, filename: str) -> str:
    generic = {"application/octet-stream", "application/force-download", "binary/octet-stream", ""}
    ct = (declared_ct or "").strip().lower()
    if ct not in generic and ct in ALLOWED_ALL_TYPES:
        return ct
    ext = Path(filename or "").suffix.lower()
    return _EXT_MIME_MAP.get(ext, ct)


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY / CASTE CONSTANTS  (NEW in v7.0)
# ══════════════════════════════════════════════════════════════════════════════

# Canonical seat type names as used in JoSAA data
CANONICAL_SEAT_TYPES = ["OPEN", "OBC-NCL", "SC", "ST", "GEN-EWS"]

# Mapping from user-friendly input to canonical seat type(s) to query
CATEGORY_SEAT_TYPE_MAP: Dict[str, List[str]] = {
    # General / Open
    "open":      ["OPEN"],
    "general":   ["OPEN"],
    "gen":       ["OPEN"],
    "unreserved":["OPEN"],
    # OBC
    "obc":       ["OBC-NCL", "OBC"],
    "obc-ncl":   ["OBC-NCL", "OBC"],
    "obc ncl":   ["OBC-NCL", "OBC"],
    # SC
    "sc":        ["SC"],
    "scheduled caste": ["SC"],
    # ST
    "st":        ["ST"],
    "scheduled tribe": ["ST"],
    # EWS
    "ews":       ["GEN-EWS", "EWS"],
    "gen-ews":   ["GEN-EWS", "EWS"],
    "gen ews":   ["GEN-EWS", "EWS"],
    "economically weaker section": ["GEN-EWS", "EWS"],
}

# Approximate rank relaxation multipliers relative to OPEN closing rank
# Used when category data is absent — gives estimated cut-off for the category
CATEGORY_RANK_RELAXATION: Dict[str, float] = {
    "OPEN":    1.0,
    "GEN-EWS": 1.15,   # EWS: ~15% more ranks eligible
    "OBC-NCL": 1.35,   # OBC: ~35% more ranks eligible
    "SC":      2.80,   # SC: cutoffs typically 2.5–3× OPEN rank
    "ST":      4.50,   # ST: cutoffs typically 4–5× OPEN rank
}

# Human-readable descriptions for the explainer endpoint
CATEGORY_DESCRIPTIONS: Dict[str, dict] = {
    "OPEN": {
        "full_name": "General / Open Category",
        "eligibility": "All candidates; no reservation benefit applied.",
        "seat_reservation_pct": "~50.5% (unreserved seats after deducting all reserved categories)",
        "rank_advantage": "No relaxation — competes on merit.",
        "jee_note": "Uses raw CRL (Common Rank List) for NITs/IIITs and JEE Adv rank for IITs.",
        "quota_note": "Eligible for AI (All India), HS (Home State), OS (Other State) quotas at NITs.",
    },
    "OBC-NCL": {
        "full_name": "Other Backward Classes – Non-Creamy Layer",
        "eligibility": "OBC certificate required; family income < ₹8 LPA (non-creamy layer).",
        "seat_reservation_pct": "27% of seats reserved",
        "rank_advantage": "Closing ranks typically 1.3–1.5× the OPEN closing rank for the same program.",
        "jee_note": "Separate OBC-NCL rank list. Annual income certificate must be ≤3 years old.",
        "quota_note": "OBC-NCL benefit applies only at Centrally Funded Technical Institutions (IITs, NITs, IIITs, CFTIs). State colleges have their own OBC rules.",
    },
    "SC": {
        "full_name": "Scheduled Caste",
        "eligibility": "Valid SC caste certificate issued by competent authority.",
        "seat_reservation_pct": "15% of seats reserved",
        "rank_advantage": "Closing ranks typically 2.5–3× the OPEN closing rank for the same program.",
        "jee_note": "Separate SC rank list. Reservation applies at all JoSAA-participating institutes.",
        "quota_note": "SC students are eligible for scholarships — check NSP and state schemes.",
    },
    "ST": {
        "full_name": "Scheduled Tribe",
        "eligibility": "Valid ST caste certificate issued by competent authority.",
        "seat_reservation_pct": "7.5% of seats reserved",
        "rank_advantage": "Closing ranks typically 4–5× the OPEN closing rank for the same program.",
        "jee_note": "Separate ST rank list. Fewer seats mean more variability year to year.",
        "quota_note": "ST students are eligible for tribal sub-plan scholarships — check state tribal welfare dept.",
    },
    "GEN-EWS": {
        "full_name": "General – Economically Weaker Section",
        "eligibility": "Open category only; family income < ₹8 LPA + no land/property above threshold.",
        "seat_reservation_pct": "10% of seats reserved (added in 2019)",
        "rank_advantage": "Closing ranks typically 1.1–1.2× the OPEN closing rank for the same program.",
        "jee_note": "EWS certificate must be ≤1 year old at time of counselling. Not applicable in all states.",
        "quota_note": "EWS benefit is only at CFTIs. State colleges have their own EWS rules.",
    },
}


def resolve_category_seat_types(category: str, available_seat_types: set) -> Tuple[List[str], bool]:
    """
    Returns (list_of_seat_types_to_query, is_exact_match).
    is_exact_match=False means we fell back to OPEN because the dataset
    doesn't have category-specific rows — caller should warn the user.
    """
    key = normalize_key(category)
    desired = CATEGORY_SEAT_TYPE_MAP.get(key, [str(category).strip()])

    matched = [st for st in desired if st in available_seat_types]
    if matched:
        return matched, True

    # Fallback: return OPEN rows and flag it
    open_types = [st for st in ["OPEN"] if st in available_seat_types]
    if not open_types and available_seat_types:
        open_types = list(available_seat_types)[:1]
    return open_types, False


def estimate_category_closing_rank(open_closing_rank: float, category: str) -> int:
    """
    When category-specific data is missing, estimate the expected closing rank
    by applying the relaxation multiplier to the OPEN closing rank.
    """
    canonical = _canonicalize_category(category)
    multiplier = CATEGORY_RANK_RELAXATION.get(canonical, 1.0)
    return int(round(open_closing_rank * multiplier))


def _canonicalize_category(category: str) -> str:
    key = normalize_key(category)
    desired = CATEGORY_SEAT_TYPE_MAP.get(key, [str(category).strip()])
    return desired[0] if desired else "OPEN"


# ══════════════════════════════════════════════════════════════════════════════
# RANK PREDICTOR CONSTANTS  (NEW in v7.0)
# ══════════════════════════════════════════════════════════════════════════════

# JEE Mains 2024 approximate rank-percentile table (interpolated from official data)
# Format: (percentile, approximate_CRL_rank)
MAINS_PERCENTILE_RANK_TABLE: List[Tuple[float, int]] = [
    (100.0,      1),
    (99.99,     20),
    (99.95,    100),
    (99.90,    200),
    (99.80,    400),
    (99.70,    600),
    (99.50,   1000),
    (99.00,   2000),
    (98.50,   3000),
    (98.00,   4000),
    (97.00,   6000),
    (96.00,   8000),
    (95.00,  10000),
    (94.00,  12000),
    (93.00,  14500),
    (92.00,  17000),
    (91.00,  20000),
    (90.00,  23000),
    (89.00,  26000),
    (88.00,  30000),
    (87.00,  34000),
    (86.00,  38000),
    (85.00,  43000),
    (84.00,  48000),
    (83.00,  54000),
    (82.00,  60000),
    (80.00,  73000),
    (78.00,  88000),
    (75.00, 110000),
    (70.00, 148000),
    (65.00, 185000),
    (60.00, 220000),
    (50.00, 290000),
]

# JEE Mains approximate marks-to-percentile mapping (300 marks total; varies by session)
MAINS_MARKS_PERCENTILE_TABLE: List[Tuple[int, float]] = [
    (300, 100.00),
    (280, 99.99),
    (260, 99.95),
    (250, 99.90),
    (240, 99.80),
    (230, 99.60),
    (220, 99.30),
    (210, 98.80),
    (200, 98.00),
    (190, 96.80),
    (180, 95.00),
    (170, 92.50),
    (160, 89.00),
    (150, 85.00),
    (140, 79.00),
    (130, 72.00),
    (120, 64.00),
    (110, 55.00),
    (100, 45.00),
    ( 90, 35.00),
    ( 80, 26.00),
    ( 70, 18.00),
    ( 60, 11.00),
    ( 50,  6.00),
    ( 40,  2.50),
    (  0,  0.00),
]

# JEE Advanced 2024 approximate marks-to-rank mapping (360 marks total)
ADV_MARKS_RANK_TABLE: List[Tuple[int, int]] = [
    (340,    1),
    (320,   10),
    (300,   50),
    (280,  150),
    (260,  350),
    (240,  700),
    (220, 1200),
    (200, 2000),
    (180, 3000),
    (160, 4500),
    (140, 6500),
    (120, 9000),
    (100,12000),
    ( 90,15000),
    ( 80,18000),
    ( 70,21000),
    ( 60,24000),
    (  0,50000),
]


def _interpolate(table: List[Tuple], x: float, x_col: int = 0, y_col: int = 1) -> float:
    """Linear interpolation between two table rows."""
    xs = [row[x_col] for row in table]
    ys = [row[y_col] for row in table]

    # Clamp
    if x >= xs[0]:  return float(ys[0])
    if x <= xs[-1]: return float(ys[-1])

    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x1 <= x <= x0:
            t = (x - x0) / (x1 - x0)
            return float(ys[i] + t * (ys[i + 1] - ys[i]))
    return float(ys[-1])


def percentile_to_rank(percentile: float) -> int:
    """Convert JEE Mains percentile to approximate CRL rank."""
    rank = _interpolate(MAINS_PERCENTILE_RANK_TABLE, percentile)
    return max(1, int(round(rank)))


def rank_to_percentile(rank: int) -> float:
    """Convert JEE Mains CRL rank to approximate percentile."""
    # Reverse lookup — rank is y_col in table (col 1), percentile is x_col (col 0)
    reversed_table = [(row[1], row[0]) for row in MAINS_PERCENTILE_RANK_TABLE]
    pct = _interpolate(reversed_table, float(rank))
    return round(min(100.0, max(0.0, pct)), 2)


def marks_to_percentile_mains(marks: float) -> float:
    """Convert JEE Mains marks (0–300) to approximate percentile."""
    pct = _interpolate(MAINS_MARKS_PERCENTILE_TABLE, marks)
    return round(max(0.0, min(100.0, pct)), 2)


def marks_to_rank_advanced(marks: float) -> int:
    """Convert JEE Advanced marks (0–360) to approximate rank."""
    rank = _interpolate(ADV_MARKS_RANK_TABLE, marks)
    return max(1, int(round(rank)))


def predict_branch_chances(student_rank: int, home_state: str, category: str) -> List[dict]:
    """
    Scan the entire cutoff dataset and return a branch-by-branch probability matrix
    for the student. Groups by branch keyword categories.
    """
    results: List[dict] = []
    canon_cat = _canonicalize_category(category)
    cat_available = canon_cat in {st.strip() for st in cutoffs_df["Seat Type"].unique()}

    # Decide which rows to use
    if cat_available:
        df_cat = cutoffs_df[cutoffs_df["Seat Type"] == canon_cat]
    else:
        df_cat = cutoffs_df[cutoffs_df["Seat Type"] == "OPEN"].copy()
        # Scale closing ranks by relaxation factor
        relax = CATEGORY_RANK_RELAXATION.get(canon_cat, 1.0)
        df_cat = df_cat.copy()
        df_cat["Closing Rank"] = (df_cat["Closing Rank"] * relax).round(0).astype(int)
        df_cat["Opening Rank"] = (df_cat["Opening Rank"] * relax).round(0).astype(int)

    df_gn = df_cat[df_cat["Gender"] == "Gender-Neutral"]

    # Group by branch bucket
    branch_buckets: Dict[str, List[float]] = {}
    for _, row in df_gn.iterrows():
        branch = str(row["Academic Program Name"])
        inst   = str(row["Institute"])
        # Apply quota filter roughly
        applicable = get_applicable_quotas(home_state, inst)
        if not any(q == str(row["Quota"]).strip() for q in applicable):
            continue
        closing = float(row["Closing Rank"])
        if closing < student_rank * 0.3:  # way out of reach even for estimation
            continue
        prob = compute_probability(student_rank, float(row["Opening Rank"]), closing, inst)
        # Bucket by interest category
        for interest_key, keywords in INTEREST_BRANCH_MAP.items():
            if interest_key == "undecided":
                continue
            bl = branch.lower()
            if any(_kw_matches(kw, bl) for kw in keywords):
                branch_buckets.setdefault(interest_key, []).append(prob)
                break
        else:
            branch_buckets.setdefault("other", []).append(prob)

    for bucket, probs in branch_buckets.items():
        if not probs:
            continue
        avg_prob = round(sum(probs) / len(probs), 1)
        max_prob = round(max(probs), 1)
        results.append({
            "branch_category": bucket,
            "avg_admission_probability": avg_prob,
            "best_case_probability": max_prob,
            "options_count": len(probs),
            "chance_label": chance_label(avg_prob),
        })

    results.sort(key=lambda x: -x["avg_admission_probability"])
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MEGA MEDIA STORE (binary files only)
# ══════════════════════════════════════════════════════════════════════════════

class MegaMediaStore:
    def __init__(self):
        self.m = None
        self._media_node_id: Optional[str] = None
        self._lock = threading.Lock()

    def connect(self):
        if not _MEGA_AVAILABLE:
            print("[MEGA] mega.py not installed — media disabled")
            return
        if not MEGA_PASSWORD:
            print("[MEGA] MEGA_PASSWORD not set — media disabled")
            return
        try:
            self.m = Mega().login(MEGA_EMAIL, MEGA_PASSWORD)
            print("[MEGA] Connected")
            self._ensure_folder()
        except Exception as e:
            print(f"[MEGA] Login failed: {e}")
            self.m = None

    def _ensure_folder(self):
        try:
            node_id = self.m.find_path_descriptor("sc_media")
            if not node_id:
                result  = self.m.create_folder("sc_media")
                node_id = result.get("sc_media") or next(iter(result.values()))
            self._media_node_id = node_id
            print(f"[MEGA] sc_media node={node_id}")
        except Exception as e:
            print(f"[MEGA] folder init failed: {e}")

    def _destroy_by_name(self, filename: str):
        try:
            result = self.m.find(filename, exclude_deleted=True)
            if result:
                self.m.destroy(result[0])
        except Exception:
            pass

    @staticmethod
    def _cleanup(path: str):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    def upload_get_link(self, staging_path: str, filename: str) -> str:
        if not self.m:
            self._cleanup(staging_path)
            raise RuntimeError("MEGA not connected — check MEGA_EMAIL / MEGA_PASSWORD")
        try:
            if not self._media_node_id:
                self._ensure_folder()
            self._destroy_by_name(filename)
            upload_resp = self.m.upload(staging_path, self._media_node_id)
            link = self.m.get_upload_link(upload_resp)
            if not link:
                raise RuntimeError("MEGA returned no public link")
            print(f"[MEGA] {filename} → {link[:60]}…")
            return link
        except Exception as e:
            raise RuntimeError(f"MEGA upload failed ({filename}): {e}") from e
        finally:
            self._cleanup(staging_path)

    def delete_file(self, filename: str):
        if not self.m:
            return
        try:
            result = self.m.find(filename, exclude_deleted=True)
            if result:
                self.m.delete(result[0])
                print(f"[MEGA] Deleted {filename}")
        except Exception as e:
            print(f"[MEGA] Delete {filename}: {e}")


_mega = MegaMediaStore()


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket, channel: str):
        await ws.accept()
        async with self._lock:
            self.connections.setdefault(channel, set()).add(ws)

    async def disconnect(self, ws: WebSocket, channel: str):
        async with self._lock:
            self.connections.get(channel, set()).discard(ws)

    async def broadcast(self, channel: str, data: dict):
        async with self._lock:
            conns = list(self.connections.get(channel, set()))
        dead = []
        for ws in conns:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                ch = self.connections.get(channel, set())
                for d in dead:
                    ch.discard(d)


ws_manager = ConnectionManager()


# ══════════════════════════════════════════════════════════════════════════════
# APP LIFESPAN
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, init_db)
    await loop.run_in_executor(_executor, db_purge_expired_groups)
    await loop.run_in_executor(_executor, _mega.connect)
    asyncio.create_task(_group_cleanup_loop())
    yield
    print("[SHUTDOWN] Goodbye.")


app = FastAPI(
    title="JEE College Advisor + StudyCord",
    version="7.0.0",
    lifespan=lifespan,
)
templates = Jinja2Templates(directory="templates")


async def _group_cleanup_loop():
    while True:
        await asyncio.sleep(3600)
        loop = asyncio.get_event_loop()
        n = await loop.run_in_executor(_executor, db_purge_expired_groups)
        if n:
            print(f"[CLEANUP] Purged {n} expired groups")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = "data"
cutoffs_df  = pd.read_excel(os.path.join(DATA_DIR, "JEE_2025_Cutoffs.xlsx"))
colleges_df = pd.read_excel(os.path.join(DATA_DIR, "JEE_2025_ALL_128_COLLEGES.xlsx"))

cutoffs_df.columns  = [c.strip() for c in cutoffs_df.columns]
colleges_df.columns = [c.strip() for c in colleges_df.columns]

REQUIRED_CUTOFF_COLUMNS = {
    "Institute", "Academic Program Name", "Quota",
    "Seat Type", "Gender", "Opening Rank", "Closing Rank",
}
missing = REQUIRED_CUTOFF_COLUMNS - set(cutoffs_df.columns)
if missing:
    raise RuntimeError(f"Cutoff file missing columns: {sorted(missing)}")

for col in ["Institute", "Academic Program Name", "Quota", "Seat Type", "Gender"]:
    cutoffs_df[col] = cutoffs_df[col].astype(str).str.strip()
for col in ["Opening Rank", "Closing Rank"]:
    cutoffs_df[col] = pd.to_numeric(cutoffs_df[col], errors="coerce")
cutoffs_df = cutoffs_df.dropna(subset=["Opening Rank", "Closing Rank"])

college_info: Dict[str, dict] = {}
for _, row in colleges_df.iterrows():
    college_info[str(row["College Name"]).strip()] = {
        "website":  str(row.get("Official Website", "")).strip(),
        "location": str(row.get("Location", "")).strip(),
        "category": str(row.get("Category", "")).strip(),
    }

INSTITUTE_QUOTAS: Dict[str, set] = {
    inst: set(grp["Quota"].dropna().astype(str).str.strip())
    for inst, grp in cutoffs_df.groupby("Institute")
}

# What seat types are actually present in the dataset
AVAILABLE_SEAT_TYPES_IN_DATA: set = set(
    cutoffs_df["Seat Type"].dropna().astype(str).str.strip().unique()
)
AVAILABLE_SEAT_TYPES: List[str] = sorted(AVAILABLE_SEAT_TYPES_IN_DATA)

# Advertise ALL possible categories to UI even if data only has OPEN
ALL_CATEGORIES_FOR_UI: List[str] = ["OPEN", "OBC-NCL", "SC", "ST", "GEN-EWS"]

QUOTA_ORDER = ["AI", "HS", "OS", "GO", "JK", "LA"]

print(f"[DATA] Loaded {len(cutoffs_df)} cutoff rows. Seat types in data: {AVAILABLE_SEAT_TYPES}")

# ══════════════════════════════════════════════════════════════════════════════
# INSTITUTION CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

_IIT_PAT  = re.compile(r'\bindian institute of technology\b|\biit\b', re.I)
_NIT_PAT  = re.compile(r'\bnational institute of technology\b|\bnit\b', re.I)
_IIIT_PAT = re.compile(r'\bindian institute of information technology\b|\biiit\b', re.I)


def classify_institute(name: str) -> str:
    n = name.strip()
    if _IIT_PAT.search(n):  return "IIT"
    if _NIT_PAT.search(n):  return "NIT"
    if _IIIT_PAT.search(n): return "IIIT"
    return "Other"


def is_iit(name: str) -> bool:
    return classify_institute(name) == "IIT"


def normalize_key(value: str) -> str:
    text = str(value or "").casefold().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ══════════════════════════════════════════════════════════════════════════════
# INTEREST → BRANCH MAP
# ══════════════════════════════════════════════════════════════════════════════

INTEREST_BRANCH_MAP: Dict[str, List[str]] = {
    "coding": [
        "computer science", "computer science and engineering", "cse", "cs",
        "information technology", "it", "software engineering", "software",
        "data science", "data engineering", "data science and engineering",
        "artificial intelligence", "ai", "artificial intelligence and data science",
        "artificial intelligence and machine learning",
        "artificial intelligence and data engineering",
        "artificial intelligence and data analytics", "machine learning",
        "computational engineering", "computational and data science", "computing",
        "informatics", "cyber security", "cybersecurity", "computer engineering",
        "vlsi", "microelectronics", "integrated circuit design", "ic design",
        "electronics and vlsi", "electronics engineering (vlsi design and technology)",
        "communication systems",
    ],
    "electronics": [
        "electrical engineering", "electrical and electronics engineering",
        "electronics engineering", "electronics and communication engineering",
        "electronics and electrical communication engineering",
        "electronics and instrumentation engineering", "instrumentation engineering",
        "instrumentation and control engineering",
        "instrumentation and biomedical engineering",
        "power and automation", "power systems",
        "electrical engineering (power and automation)",
        "electrical engineering (ic design and technology)",
        "electrical engineering (integrated circuit design and technology)",
        "integrated circuit design", "vlsi design", "microelectronics", "embedded systems",
    ],
    "research": [
        "engineering physics", "engineering science", "physics", "applied physics",
        "mathematics", "mathematics and computing", "mathematics & computing",
        "mathematics and scientific computing", "mathematics and data science",
        "mathematics and computing technology", "computational mathematics",
        "applied mathematics", "statistics", "statistics and data science",
        "scientific computing", "computational science", "chemistry",
        "chemical sciences", "chemical science", "physics with specialization",
        "chemistry with specialization", "earth sciences", "biological science",
        "biosciences", "interdisciplinary sciences",
    ],
    "mba": [
        "industrial engineering", "industrial engineering and operations research",
        "industrial and systems engineering", "industrial and production engineering",
        "production engineering", "production and industrial engineering",
        "manufacturing science", "manufacturing science and engineering",
        "operations research", "management", "economics", "industrial management",
        "logistics", "supply chain", "digital business", "hospital management",
        "healthcare management", "mba", "dual degree mba", "engineering design",
    ],
    "core_engg": [
        "mechanical engineering", "mechatronics", "mechatronics engineering",
        "manufacturing", "civil engineering", "civil and infrastructure engineering",
        "structural engineering", "geotechnical engineering", "electrical engineering",
        "electronics", "chemical engineering", "chemical and biochemical engineering",
        "chemical science and technology", "metallurgical engineering",
        "metallurgical and materials engineering", "materials engineering",
        "materials science", "materials science and engineering",
        "materials science and metallurgical engineering", "material science",
        "polymer", "ceramic engineering", "textile technology", "mining engineering",
        "mining machinery engineering", "mineral engineering",
        "mineral and metallurgical engineering", "petroleum engineering",
        "aerospace engineering", "ocean engineering", "naval architecture",
        "naval architecture and ocean engineering",
        "ocean engineering and naval architecture", "agricultural engineering",
        "agricultural and food engineering", "environmental engineering",
        "environmental science and engineering", "energy engineering",
        "biotechnology", "bio technology", "bioengineering", "bio engineering",
        "biological engineering", "biomedical engineering", "bio medical engineering",
        "biotechnology and biochemical engineering", "biotechnology and bioinformatics",
        "biosciences and bioengineering", "biochemical engineering",
        "industrial chemistry", "pharmaceutical engineering",
        "pharmaceutical engineering & technology", "design", "industrial design",
        "architecture", "planning", "general engineering",
        "engineering and computational mechanics",
    ],
    "earth_science": [
        "applied geology", "applied geophysics", "exploration geophysics",
        "earth sciences", "geological technology", "geophysical technology",
        "geology", "geophysics",
    ],
    "science": [
        "biology", "biological science", "physics", "chemistry", "mathematics",
        "economics", "statistics", "physical science", "chemical science",
        "interdisciplinary sciences",
    ],
    "architecture_design": [
        "architecture", "planning", "design", "industrial design",
        "engineering design", "bachelor of architecture", "bachelor of planning",
    ],
    "undecided": [],
}


def _kw_matches(kw: str, text: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])"
    return bool(re.search(pattern, text))


def branch_matches_interests(branch: str, interests: List[str]) -> bool:
    if not interests or "undecided" in interests:
        return True
    bl = branch.lower()
    for interest in interests:
        if any(_kw_matches(kw, bl) for kw in INTEREST_BRANCH_MAP.get(interest, [])):
            return True
    return False


def score_branch_for_interests(branch: str, interests: List[str]) -> int:
    if not interests or "undecided" in interests:
        return 0
    bl = branch.lower()
    score = 0
    for interest in interests:
        for kw in INTEREST_BRANCH_MAP.get(interest, []):
            if _kw_matches(kw, bl):
                score += 10
    return score


# ══════════════════════════════════════════════════════════════════════════════
# STATE / QUOTA TABLES
# ══════════════════════════════════════════════════════════════════════════════

STATE_TO_DISPLAY: Dict[str, str] = {
    "andhra pradesh": "Andhra Pradesh", "arunachal pradesh": "Arunachal Pradesh",
    "assam": "Assam", "bihar": "Bihar", "chhattisgarh": "Chhattisgarh",
    "goa": "Goa", "gujarat": "Gujarat", "haryana": "Haryana",
    "himachal pradesh": "Himachal Pradesh", "jammu and kashmir": "Jammu and Kashmir",
    "jharkhand": "Jharkhand", "karnataka": "Karnataka", "kerala": "Kerala",
    "madhya pradesh": "Madhya Pradesh", "maharashtra": "Maharashtra",
    "manipur": "Manipur", "meghalaya": "Meghalaya", "mizoram": "Mizoram",
    "nagaland": "Nagaland", "odisha": "Odisha", "punjab": "Punjab",
    "rajasthan": "Rajasthan", "sikkim": "Sikkim", "tamil nadu": "Tamil Nadu",
    "telangana": "Telangana", "tripura": "Tripura", "uttar pradesh": "Uttar Pradesh",
    "uttarakhand": "Uttarakhand", "west bengal": "West Bengal", "delhi": "Delhi",
    "ladakh": "Ladakh", "andaman and nicobar islands": "Andaman and Nicobar Islands",
    "chandigarh": "Chandigarh",
    "dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "lakshadweep": "Lakshadweep", "puducherry": "Puducherry",
}

STATE_ALIASES: Dict[str, str] = {
    "andaman nicobar": "andaman and nicobar islands",
    "andaman and nicobar": "andaman and nicobar islands",
    "dadra nagar haveli daman diu": "dadra and nagar haveli and daman and diu",
    "dadra and nagar haveli": "dadra and nagar haveli and daman and diu",
    "daman and diu": "dadra and nagar haveli and daman and diu",
    "j and k": "jammu and kashmir", "jk": "jammu and kashmir",
    "jammu kashmir": "jammu and kashmir", "nct delhi": "delhi",
    "new delhi": "delhi", "orissa": "odisha", "pondicherry": "puducherry",
}

STATE_NIT_MAP: Dict[str, tuple] = {
    "andhra pradesh":    ("National Institute of Technology, Andhra Pradesh",),
    "arunachal pradesh": ("National Institute of Technology Arunachal Pradesh",),
    "assam":             ("National Institute of Technology, Silchar",),
    "bihar":             ("National Institute of Technology Patna",),
    "chhattisgarh":      ("National Institute of Technology Raipur",),
    "delhi":             ("National Institute of Technology Delhi",),
    "goa":               ("National Institute of Technology Goa",),
    "gujarat":           ("Sardar Vallabhbhai National Institute of Technology, Surat",),
    "haryana":           ("National Institute of Technology, Kurukshetra",),
    "himachal pradesh":  ("National Institute of Technology Hamirpur",),
    "jammu and kashmir": ("National Institute of Technology, Srinagar",),
    "jharkhand":         ("National Institute of Technology, Jamshedpur",),
    "karnataka":         ("National Institute of Technology Karnataka, Surathkal",),
    "kerala":            ("National Institute of Technology Calicut",),
    "ladakh":            ("National Institute of Technology, Srinagar",),
    "madhya pradesh":    ("Maulana Azad National Institute of Technology Bhopal",),
    "maharashtra":       ("Visvesvaraya National Institute of Technology, Nagpur",),
    "manipur":           ("National Institute of Technology, Manipur",),
    "meghalaya":         ("National Institute of Technology Meghalaya",),
    "mizoram":           ("National Institute of Technology, Mizoram",),
    "nagaland":          ("National Institute of Technology Nagaland",),
    "odisha":            ("National Institute of Technology, Rourkela",),
    "puducherry":        ("National Institute of Technology Puducherry",),
    "punjab":            ("Dr. B R Ambedkar National Institute of Technology, Jalandhar",),
    "rajasthan":         ("Malaviya National Institute of Technology Jaipur",),
    "sikkim":            ("National Institute of Technology Sikkim",),
    "tamil nadu":        ("National Institute of Technology, Tiruchirappalli",),
    "telangana":         ("National Institute of Technology, Warangal",),
    "tripura":           ("National Institute of Technology Agartala",),
    "uttar pradesh":     ("Motilal Nehru National Institute of Technology Allahabad",),
    "uttarakhand":       ("National Institute of Technology, Uttarakhand",),
    "west bengal":       ("National Institute of Technology Durgapur",),
}

OTHER_HOME_STATE_INST_MAP: Dict[str, tuple] = {
    "assam":             ("Assam University, Silchar",),
    "bihar":             ("Birla Institute of Technology, Patna Off-Campus",),
    "chandigarh":        ("Punjab Engineering College, Chandigarh",),
    "jammu and kashmir": ("Islamic University of Science and Technology Kashmir",),
    "jharkhand": (
        "Birla Institute of Technology, Deoghar Off-Campus",
        "Birla Institute of Technology, Mesra, Ranchi",
    ),
    "odisha": ("Institute of Chemical Technology, Mumbai: Indian Oil Odisha Campus, Bhubaneswar",),
    "puducherry": ("Puducherry Technological University, Puducherry",),
    "west bengal": (
        "Ghani Khan Choudhary Institute of Engineering and Technology, Malda, West Bengal",
        "Indian Institute of Engineering Science and Technology, Shibpur",
    ),
}

SPECIAL_QUOTAS_BY_STATE: Dict[str, tuple] = {
    "goa":               ("GO",),
    "jammu and kashmir": ("JK",),
    "ladakh":            ("LA",),
}

HOME_STATE_INST_PATTERNS: Dict[str, tuple] = {}
for _sk, _names in STATE_NIT_MAP.items():
    HOME_STATE_INST_PATTERNS.setdefault(_sk, ())
    HOME_STATE_INST_PATTERNS[_sk] += tuple(normalize_key(n) for n in _names)
for _sk, _names in OTHER_HOME_STATE_INST_MAP.items():
    HOME_STATE_INST_PATTERNS.setdefault(_sk, ())
    HOME_STATE_INST_PATTERNS[_sk] += tuple(normalize_key(n) for n in _names)

STATE_NEIGHBORS: Dict[str, tuple] = {
    "andhra pradesh":   ("telangana", "tamil nadu", "karnataka", "odisha"),
    "arunachal pradesh": ("assam", "nagaland"),
    "assam":            ("arunachal pradesh", "nagaland", "mizoram", "meghalaya", "tripura", "west bengal"),
    "bihar":            ("uttar pradesh", "jharkhand", "west bengal"),
    "chhattisgarh":     ("uttar pradesh", "bihar", "odisha", "andhra pradesh", "telangana", "maharashtra", "madhya pradesh"),
    "goa":              ("karnataka", "maharashtra"),
    "gujarat":          ("rajasthan", "madhya pradesh", "maharashtra", "dadra and nagar haveli and daman and diu"),
    "haryana":          ("punjab", "rajasthan", "uttarakhand", "delhi", "himachal pradesh"),
    "himachal pradesh": ("jammu and kashmir", "ladakh", "punjab", "haryana", "uttarakhand"),
    "jammu and kashmir": ("ladakh", "himachal pradesh", "punjab"),
    "jharkhand":        ("bihar", "west bengal", "odisha", "andhra pradesh", "chhattisgarh"),
    "karnataka":        ("goa", "maharashtra", "telangana", "andhra pradesh", "tamil nadu", "kerala"),
    "kerala":           ("tamil nadu",),
    "madhya pradesh":   ("uttar pradesh", "chhattisgarh", "maharashtra", "gujarat", "rajasthan"),
    "maharashtra":      ("gujarat", "madhya pradesh", "chhattisgarh", "telangana", "karnataka", "goa"),
    "manipur":          ("nagaland", "mizoram", "assam"),
    "meghalaya":        ("assam",),
    "mizoram":          ("assam", "tripura", "meghalaya", "manipur"),
    "nagaland":         ("assam", "arunachal pradesh", "mizoram", "manipur"),
    "odisha":           ("west bengal", "jharkhand", "chhattisgarh", "andhra pradesh"),
    "punjab":           ("jammu and kashmir", "himachal pradesh", "haryana", "rajasthan"),
    "rajasthan":        ("punjab", "haryana", "gujarat", "madhya pradesh", "uttar pradesh"),
    "sikkim":           ("west bengal",),
    "tamil nadu":       ("kerala", "karnataka", "andhra pradesh", "puducherry"),
    "telangana":        ("maharashtra", "chhattisgarh", "odisha", "andhra pradesh", "karnataka"),
    "tripura":          ("assam", "mizoram"),
    "uttarakhand":      ("himachal pradesh", "haryana", "uttar pradesh"),
    "uttar pradesh":    ("uttarakhand", "himachal pradesh", "haryana", "delhi", "rajasthan", "madhya pradesh", "chhattisgarh", "bihar", "jharkhand"),
    "west bengal":      ("sikkim", "assam", "bihar", "jharkhand", "odisha", "tripura"),
    "andaman and nicobar islands": (),
    "chandigarh":       ("punjab", "haryana"),
    "dadra and nagar haveli and daman and diu": ("gujarat", "maharashtra"),
    "delhi":            ("haryana", "uttar pradesh", "rajasthan"),
    "ladakh":          ("jammu and kashmir", "himachal pradesh"),
    "lakshadweep":      (),
    "puducherry":       ("tamil nadu", "andhra pradesh"),
}

SEAT_TYPE_ALIASES: Dict[str, tuple] = {
    "general": ("OPEN",), "gen": ("OPEN",), "open": ("OPEN",),
    "ews": ("GEN-EWS", "EWS"), "gen ews": ("GEN-EWS", "EWS"),
    "obc": ("OBC-NCL", "OBC"), "obc ncl": ("OBC-NCL", "OBC"),
    "sc": ("SC",), "st": ("ST",),
}

# ══════════════════════════════════════════════════════════════════════════════
# RANKING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

_SAFE_MIN_PROB     = 85.0
_GOOD_MIN_PROB     = 60.0
_MODERATE_MIN_PROB = 35.0
_IIT_VOLATILITY    = 0.08
_NIT_VOLATILITY    = 0.14
_OTHER_VOLATILITY  = 0.18


def _volatility(institute: str) -> float:
    cat = classify_institute(institute)
    if cat == "IIT":  return _IIT_VOLATILITY
    if cat == "NIT":  return _NIT_VOLATILITY
    return _OTHER_VOLATILITY


def compute_probability(student_rank: int, opening_rank: float,
                        closing_rank: float, institute: str = "") -> float:
    if student_rank <= opening_rank:
        margin = max(opening_rank - student_rank, 0)
        boost  = min(margin / max(opening_rank, 1) * 10, 4.0)
        return round(min(95.0 + boost, 99.0), 1)
    rank_range = max(closing_rank - opening_rank, 1.0)
    vol_buffer = closing_rank * _volatility(institute)
    window     = rank_range + vol_buffer
    z          = (student_rank - closing_rank) / window
    prob       = 100.0 / (1.0 + math.exp(3.0 * z))
    prob       = min(max(prob, 50.0 if student_rank >= closing_rank else 35.0), 94.0)
    return round(prob, 1)


def chance_label(prob: float) -> str:
    if prob >= _SAFE_MIN_PROB:     return "Safe"
    if prob >= _GOOD_MIN_PROB:     return "Good"
    if prob >= _MODERATE_MIN_PROB: return "Moderate"
    if prob >= 15.0:               return "Reach"
    return "Difficult"


def canonical_state(home_state: str) -> str:
    sk = normalize_key(home_state)
    return STATE_ALIASES.get(sk, sk)


def _find_state_from_location(location: str) -> Optional[str]:
    if not location:
        return None
    text = normalize_key(location)
    for state in STATE_TO_DISPLAY.keys():
        if state in text:
            return state
    for alias, canonical_val in STATE_ALIASES.items():
        if alias in text:
            return canonical_val
    return None


def _state_distance_rank(home_state: str, college_state: Optional[str]) -> int:
    home = canonical_state(home_state)
    if not college_state or not home:
        return 3
    if college_state == home:
        return 0
    neighbors = STATE_NEIGHBORS.get(home, ())
    if college_state in neighbors:
        return neighbors.index(college_state) + 1
    return 2


def _is_home_state_institute(state_key: str, institute: str) -> bool:
    ik = normalize_key(institute)
    return any(p in ik for p in HOME_STATE_INST_PATTERNS.get(state_key, ()))


def get_selected_seat_types(profile) -> Tuple[List[str], bool]:
    """
    Returns (seat_types_to_query, is_exact_category_match).
    When category data is absent, returns OPEN rows + is_exact=False.
    """
    category = str(profile.seat_type or profile.category or "OPEN").strip()
    seat_types, is_exact = resolve_category_seat_types(category, AVAILABLE_SEAT_TYPES_IN_DATA)
    return seat_types, is_exact


def ordered_quotas(quotas) -> List[str]:
    qs = {str(q).strip() for q in quotas if str(q).strip()}
    return [q for q in QUOTA_ORDER if q in qs] + sorted(qs - set(QUOTA_ORDER))


def get_applicable_quotas(home_state: str, institute: str,
                          selected_quota: str = "") -> List[str]:
    available = INSTITUTE_QUOTAS.get(str(institute), set())
    if not available:
        return ["AI"]
    sq = str(selected_quota or "").strip().upper()
    if sq:
        return [sq] if sq in available else []
    if is_iit(institute):
        return ["AI"] if "AI" in available else sorted(available)
    if available == {"AI"}:
        return ["AI"]
    state_key = canonical_state(home_state)
    quotas: List[str] = []
    for sq in SPECIAL_QUOTAS_BY_STATE.get(state_key, ()):
        if sq in available:
            quotas.append(sq)
    if _is_home_state_institute(state_key, institute) and "HS" in available:
        quotas.append("HS")
    if quotas:
        return list(dict.fromkeys(quotas))
    if "OS" in available:
        return ["OS"]
    if "AI" in available:
        return ["AI"]
    return ordered_quotas(available)


def _rank_for(profile, institute: str) -> Optional[int]:
    if is_iit(institute):
        return profile.jee_adv_rank
    return profile.jee_mains_rank


def filter_options(profile) -> Tuple[pd.DataFrame, bool]:
    """
    Returns (filtered_df, category_data_was_exact_match).
    When category data is absent, rows come from OPEN with scaled closing ranks.
    """
    if not profile.jee_mains_rank and not profile.jee_adv_rank:
        return pd.DataFrame(), True

    seat_types, is_exact = get_selected_seat_types(profile)
    canon_cat = _canonicalize_category(str(profile.seat_type or profile.category or "OPEN"))
    relax = CATEGORY_RANK_RELAXATION.get(canon_cat, 1.0) if not is_exact else 1.0

    df = cutoffs_df.copy()
    df = df[df["Seat Type"].isin(seat_types)]

    # If using OPEN as proxy, scale the rank windows
    if not is_exact and relax != 1.0:
        df = df.copy()
        df["Closing Rank"] = (df["Closing Rank"] * relax).round(0).astype(int)
        df["Opening Rank"] = (df["Opening Rank"] * relax).round(0).astype(int)

    gender_vals = ["Gender-Neutral"]
    if str(profile.gender).lower() in ("female", "f"):
        gender_vals.append("Female-only (including Supernumerary)")
    df = df[df["Gender"].isin(gender_vals)]

    has_adv   = bool(profile.jee_adv_rank)
    has_mains = bool(profile.jee_mains_rank)
    df = df[df["Institute"].apply(lambda inst: has_adv if is_iit(inst) else has_mains)]
    if df.empty:
        return pd.DataFrame(), is_exact

    if profile.interests and "undecided" not in profile.interests:
        df = df[df["Academic Program Name"].apply(
            lambda b: branch_matches_interests(b, profile.interests)
        )]
    if df.empty:
        return pd.DataFrame(), is_exact

    results = []
    for inst, inst_df in df.groupby("Institute", sort=False):
        quotas = get_applicable_quotas(profile.home_state, inst, profile.quota)
        sub    = inst_df[inst_df["Quota"].isin(quotas)]
        if not sub.empty:
            results.append(sub)

    if not results:
        return pd.DataFrame(), is_exact

    filtered = pd.concat(results, ignore_index=True)

    def get_rank(inst: str) -> Optional[float]:
        r = _rank_for(profile, str(inst))
        return float(r) if r else None

    filtered["_student_rank"] = filtered["Institute"].apply(get_rank)
    filtered = filtered[filtered["_student_rank"].notna()]
    filtered = filtered[filtered["_student_rank"] <= filtered["Closing Rank"]]
    if filtered.empty:
        return pd.DataFrame(), is_exact

    def enrich(row):
        info = college_info.get(str(row["Institute"]).strip(), {})
        row["_website"]   = info.get("website",  "")
        row["_location"]  = info.get("location", "")
        row["_category"]  = info.get("category", "") or classify_institute(str(row["Institute"]))
        row["_inst_type"] = classify_institute(str(row["Institute"]))
        row["_is_iit"]    = row["_inst_type"] == "IIT"
        return row

    return filtered.apply(enrich, axis=1), is_exact


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gen_code() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def _state_options() -> List[str]:
    return [v for _, v in sorted(STATE_TO_DISPLAY.items(), key=lambda x: x[1])]


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════

class StudentProfile(BaseModel):
    jee_mains_rank: Optional[int] = None
    jee_adv_rank:   Optional[int] = None
    gender:         str = "male"
    home_state:     str = ""
    nearest_only:   bool = False
    interests:      List[str] = Field(default_factory=list)
    category:       str = "OPEN"     # caste/reservation category
    seat_type:      str = "OPEN"     # same as category (UI may send either)
    quota:          str = ""


class RankPredictorInput(BaseModel):
    """Input for the advanced rank predictor."""
    # JEE Mains
    mains_percentile:    Optional[float] = None   # 0–100
    mains_marks:         Optional[float] = None   # 0–300
    mains_rank:          Optional[int]   = None   # CRL rank
    # JEE Advanced
    adv_marks:           Optional[float] = None   # 0–360
    adv_rank:            Optional[int]   = None
    # Context
    gender:              str = "male"
    home_state:          str = ""
    category:            str = "OPEN"
    interests:           List[str] = Field(default_factory=list)


class CollegeCompareRequest(BaseModel):
    """Compare 2–4 colleges side by side."""
    college_names: List[str]
    branch:        str = ""
    student_rank:  Optional[int] = None
    category:      str = "OPEN"


class ChatMessage(BaseModel):
    username:        str = "guest"
    message:         str
    history:         List[dict] = Field(default_factory=list)
    student_profile: Optional[StudentProfile] = None
    college_name:    Optional[str] = None
    college_branch:  str = ""
    college_list:    List[dict] = Field(default_factory=list)
    use_perplexity:  bool = False


class CollegeDetailRequest(BaseModel):
    college_name: str
    branch:       str = ""
    use_scraper:  bool = False


class PostIn(BaseModel):
    author:       str
    avatar_color: str = "#5865F2"
    content:      str = ""
    media_url:    str = ""
    media_type:   str = ""


class ReactIn(BaseModel):
    author: str
    emoji:  str


class CommentIn(BaseModel):
    author:       str
    avatar_color: str = "#5865F2"
    content:      str = ""
    media_url:    str = ""
    media_type:   str = ""


class GroupCreateIn(BaseModel):
    name:          str
    subject:       str = ""
    expiry_months: int = 3
    group_type:    str = "public"
    created_by:    str
    avatar_color:  str = "#5865F2"


class GroupJoinIn(BaseModel):
    code:         str
    username:     str
    avatar_color: str = "#5865F2"


class GroupLeaveIn(BaseModel):
    username: str


class MsgIn(BaseModel):
    author:       str
    avatar_color: str = "#5865F2"
    content:      str = ""
    media_url:    str = ""
    media_type:   str = ""


class AiTaskIn(BaseModel):
    username: str
    subject:  str
    context:  str = ""


class CollegeEnquiryIn(BaseModel):
    college_name: str
    branch:       str = ""
    question:     str = ""


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT ENGINEERING — Groq LLM helpers
# ══════════════════════════════════════════════════════════════════════════════

def _format_college_list_for_prompt(colleges: List[dict], max_items: int = 60) -> str:
    if not colleges:
        return "No shortlisted colleges provided."

    total = len(colleges)
    lines = [f"Total shortlisted colleges: {total}"]
    for idx, college in enumerate(colleges[:max_items], start=1):
        inst = college.get("institute") or college.get("college") or "Unknown"
        prog = college.get("program") or college.get("branch") or ""
        quota = college.get("quota", "")
        seat_type = college.get("seat_type", "")
        gender = college.get("gender", "")
        opening_rank = college.get("opening_rank", "?")
        closing_rank = college.get("closing_rank", "?")
        student_rank = college.get("student_rank_used", "?")
        rank_source = college.get("rank_source", "")
        chance = college.get("chance", "")
        chance_probability = college.get("chance_probability", "")
        pref_score = college.get("preference_score", "")
        safe = college.get("in_safe_zone")
        state = college.get("college_state", "")
        inst_type = college.get("inst_type", "")
        location = college.get("location", "")
        lines.append(
            f"{idx}. {inst} — {prog} | {inst_type} | quota: {quota} | seat_type: {seat_type} | "
            f"gender: {gender} | opening: {opening_rank} | closing: {closing_rank} | "
            f"student_rank: {student_rank} ({rank_source}) | chance: {chance} | "
            f"chance_prob: {chance_probability} | pref_score: {pref_score} | "
            f"safe: {safe} | state: {state} | location: {location}"
        )
    if total > max_items:
        lines.append(f"... and {total - max_items} more shortlisted colleges not listed here.")
    return "\n".join(lines)


def _format_perplexity_tool_data(perplexity_data: dict) -> str:
    if not perplexity_data:
        return ""
    raw = str(perplexity_data.get("raw_text", "")).strip()
    query = perplexity_data.get("query", "")
    if not raw:
        return ""
    blocks = [
        "## EXTERNAL REAL-TIME DATA SOURCE: PERPLEXITY AI",
        "Use the following live Perplexity AI data as an external knowledge tool and trusted database for current college information. If this data conflicts with internal statistics or old information, prefer the Perplexity content but note any uncertainty.",
    ]
    if query:
        blocks.append(f"Query used: {query}")
    blocks.append(raw[:2500])
    return "\n\n".join(blocks)


def _build_counselor_system(profile: Optional[StudentProfile],
                            options_summary: str,
                            category_note: str = "",
                            perplexity_data: Optional[dict] = None,
                            college_list: Optional[List[dict]] = None) -> str:
    """
    Detailed, structured system prompt for the JEE counselor chat.
    v7.0: Adds caste/category awareness and quota guidance.
    """
    rank_info = ""
    if profile:
        if profile.jee_adv_rank:
            rank_info += f"- JEE Advanced Rank: {profile.jee_adv_rank}\n"
        if profile.jee_mains_rank:
            rank_info += f"- JEE Mains Rank (CRL): {profile.jee_mains_rank}\n"
        rank_info += f"- Gender: {profile.gender}\n"
        rank_info += f"- Home State: {profile.home_state or 'Not specified'}\n"
        cat = profile.seat_type or profile.category or "OPEN"
        rank_info += f"- Category / Reservation: {cat}\n"
        if cat.upper() not in ("OPEN", ""):
            desc = CATEGORY_DESCRIPTIONS.get(cat.upper(), {})
            if desc:
                rank_info += (
                    f"  → {desc.get('full_name', cat)}: {desc.get('seat_reservation_pct', '')} reserved. "
                    f"{desc.get('rank_advantage', '')}\n"
                )
        rank_info += f"- Academic Interests: {', '.join(profile.interests) if profile.interests else 'Open / Undecided'}\n"

    prompt = f"""You are an expert JEE college counselor with 20+ years of experience advising
students across IITs, NITs, IIITs, and other top engineering colleges in India.

## STUDENT PROFILE
{rank_info if rank_info else "No rank provided yet — ask the student."}

## SHORTLISTED OPTIONS (sample, top results from our database)
{options_summary if options_summary else "Not yet computed. Ask student for rank/state/category."}

{f"## CATEGORY / DATA NOTE{chr(10)}{category_note}" if category_note else ""}
"""

    if college_list:
        prompt += "\n\n## SHORTLISTED COLLEGE OPTIONS\n"
        prompt += _format_college_list_for_prompt(college_list)

    prompt += """
## YOUR ROLE & RULES
1. **Accuracy first**: Only recommend colleges where student rank ≤ closing rank
   (after applying category rank relaxation if applicable).
   - Rank < opening rank → "Safe" pick
   - Rank between opening and closing → "Variable / Moderate" risk
   - Rank > closing rank → DO NOT suggest this option
2. **IIT vs NIT distinction**: IITs use JEE Advanced rank. NITs/IIITs use JEE Mains CRL.
   Never mix them up.
3. **Quota logic**: For NITs, explain Home State (HS) vs Other State (OS) quota clearly.
   HS quota is only for the NIT in the student's home state. All other NITs use OS quota.
4. **Category / Caste guidance**: Always explain how the student's category affects
   their eligibility. Caste-based reservation applies ONLY at CFTIs (Central govt colleges
   — IITs, NITs, IIITs). State colleges have their own rules.
   - OPEN: Pure merit. No extra seats.
   - OBC-NCL: 27% reserved; closing ranks ~1.3–1.5× OPEN rank.
   - SC: 15% reserved; closing ranks ~2.5–3× OPEN rank.
   - ST: 7.5% reserved; closing ranks ~4–5× OPEN rank.
   - GEN-EWS: 10% reserved (since 2019); closing ranks ~1.1–1.2× OPEN rank.
5. **Placement data**: Give realistic avg/median/highest packages for each college.
   Mention top recruiters by name if known.
6. **Branch guidance**: Align branch recommendations with student's stated interests.
   Explain trade-offs (e.g., CSE at lower NIT vs Core Engg at top NIT).
7. **Tone**: Warm, encouraging, and realistic. Never give false hope.
   Acknowledge uncertainty where it exists.
8. **Format**: Use markdown headers, bullet points, and tables where helpful.
   Keep responses concise but complete.
9. **Context awareness**: Remember the conversation history. Don't repeat yourself.
   Build on previous messages.

## EXTERNAL DATA SOURCE GUIDELINES
Use any live Perplexity AI content provided below as an external database/tool. Treat it as a trusted source for current college admissions, placements, infrastructure, and student life details. Prefer it over outdated static data when there is a conflict, but be transparent about uncertainty.

## CHAIN OF THOUGHT
Before answering, internally:
(a) Identify what the student is really asking.
(b) Cross-check any college/rank claims against the shortlisted options above.
(c) If category ≠ OPEN and data is estimated, clarify this is approximate.
(d) Structure: specific options → reasoning → actionable next steps.

Always end with a clear next-step suggestion or question to help the student decide."""

    if perplexity_data and perplexity_data.get("raw_text"):
        prompt += "\n\n" + _format_perplexity_tool_data(perplexity_data)
    return prompt


def _extract_college_name_from_message(message: str) -> Optional[str]:
    if not message:
        return None
    patterns = [
        r"tell me about\s+(.+)",
        r"what can you tell me about\s+(.+)",
        r"about\s+(.+?)\s*(?:placements|culture|best branches|should i go here|details?|\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r"\s+(?:placements|culture|best branches|should i go here|details?)$", "", extracted, flags=re.IGNORECASE).strip()
            return extracted
    return None


def _build_college_insight_prompt(name: str, category: str, location: str,
                                  programs: List[dict],
                                  scrape_data: Optional[dict] = None,
                                  student_category: str = "OPEN") -> str:
    """
    Prompt-engineered for rich, structured college insights.
    v7.0: Adds category-specific seat info.
    """
    prog_sample = ""
    if programs:
        top_progs = programs[:8]
        prog_sample = "\n".join(
            f"  - {p['program']} | Closing Rank: {p['closing_rank']} "
            f"(Seat Type: {p['seat_type']}, Quota: {p['quota']})"
            for p in top_progs
        )

    scrape_section = ""
    if scrape_data and scrape_data.get("raw_text"):
        scrape_section = f"""
## LIVE DATA (scraped from Perplexity AI — use this to enrich your answer)
{scrape_data['raw_text'][:2000]}
"""

    cat_note = ""
    if student_category.upper() not in ("OPEN", ""):
        desc = CATEGORY_DESCRIPTIONS.get(student_category.upper(), {})
        relax = CATEGORY_RANK_RELAXATION.get(student_category.upper(), 1.0)
        cat_note = f"""
## STUDENT CATEGORY: {student_category}
{desc.get('full_name', student_category)} — {desc.get('seat_reservation_pct', 'N/A')} reserved.
Closing ranks for this category are approximately {relax:.1f}× the OPEN closing rank.
{desc.get('quota_note', '')}
"""

    return f"""You are a JEE college expert providing a comprehensive profile for a student.

## COLLEGE: {name}
- Category: {category}
- Location: {location}
- Institution Type: {classify_institute(name)}
{cat_note}

## TOP PROGRAMMES (from JEE 2025 cutoff data)
{prog_sample}
{scrape_section}

## YOUR TASK
Write a detailed college profile (400–500 words) covering these 7 sections.
Use the live data above if provided — it has real, current information.
If student category is non-OPEN, add a note about expected category-wise closing ranks.

### 1. 🏛 Reputation & NIRF Ranking
Give NIRF rank, historical ranking trend, and global recognition.

### 2. 💼 Placements (2023–24)
- Average package, Median package, Highest package
- Top 3–5 recruiters by name
- % students placed
- Sectors (IT, Core, Finance, etc.)

### 3. 🎓 Best Branches to Choose
List top 3 branches with reasons (demand, placements, research scope).

### 4. 🏕 Campus Life
Hostels, sports, cultural fest names, technical fest names, clubs.

### 5. 🔬 Research & Innovation
Labs, funded projects, patents, startup ecosystem.

### 6. 🌟 Notable Alumni
Name 2–3 well-known alumni with their current role.

### 7. ✅ Pros & Cons
3 pros and 3 cons specific to this college, not generic.

Be specific with numbers. Write in second person ("You will find…").
End with a one-sentence verdict for a JEE student."""


def _build_enquiry_prompt(college_name: str, branch: str,
                          question: str, scrape_data: Optional[dict]) -> str:
    context = ""
    if scrape_data and scrape_data.get("raw_text"):
        context = f"\n## LIVE CONTEXT (Perplexity AI data)\n{scrape_data['raw_text'][:3000]}\n"

    branch_clause = f" — {branch}" if branch else ""
    q_clause      = f'\n\nStudent Question: "{question}"' if question else ""

    return f"""You are a JEE admissions expert. A student is asking about:
**College**: {college_name}{branch_clause}
{context}

## YOUR TASK
Answer the student's question thoroughly using the context above plus your knowledge.
If the context doesn't cover it, say so honestly and provide what you know.

Structure your answer with:
1. **Direct answer** to the question (2–3 sentences)
2. **Supporting details** (bullet points, numbers, examples)
3. **Related advice** the student should know (1–2 sentences)
{q_clause}

Be specific, factual, and encouraging. Use ₹ for INR amounts."""


def _build_college_compare_prompt(colleges: List[dict], branch: str,
                                   student_rank: Optional[int],
                                   category: str) -> str:
    college_lines = "\n".join(
        f"- {c['name']} ({c['inst_type']}, {c['location']})"
        for c in colleges
    )
    rank_line = f"Student Rank: {student_rank}" if student_rank else "Rank not provided"
    cat_desc  = CATEGORY_DESCRIPTIONS.get(category.upper(), {}).get("full_name", category)

    return f"""You are a JEE college counselor. A student wants to compare these colleges:
{college_lines}

Context:
- {rank_line}
- Category: {category} ({cat_desc})
- Branch of interest: {branch or 'Not specified'}

## YOUR TASK
Create a comprehensive side-by-side comparison table and narrative covering:

### 📊 Quick Comparison Table
| Parameter | {' | '.join(c['name'].split(',')[0] for c in colleges)} |
|-----------|{'|'.join(['---'] * len(colleges))}|
| NIRF Rank | ... |
| Avg Package (₹ LPA) | ... |
| Highest Package (₹ LPA) | ... |
| Top Recruiter | ... |
| Campus Size | ... |
| Research Score | ... |

### 🏆 Ranking by Category
For each of these, rank the colleges 1st to {len(colleges)}th:
1. **Placements** — explain why
2. **Research & Academics** — explain why
3. **Campus Life & Infrastructure** — explain why
4. **Location & Industry Proximity** — explain why
5. **Value for the Branch ({branch or 'general engineering'})** — explain why

### 💡 My Recommendation
Given the student's rank and category ({category}), give a clear verdict:
which college to prefer and why. Be specific.

Use ₹ for INR. Be concise but complete."""


def _build_rank_predictor_prompt(inputs: RankPredictorInput,
                                  computed: dict) -> str:
    """Build the AI narrative for rank prediction results."""
    cat = inputs.category.upper()
    cat_desc = CATEGORY_DESCRIPTIONS.get(cat, {}).get("full_name", cat)
    relax = CATEGORY_RANK_RELAXATION.get(cat, 1.0)

    return f"""You are a JEE rank analysis expert. A student has provided their scores and wants
a comprehensive rank analysis and college prediction.

## INPUT DATA
- JEE Mains Percentile: {inputs.mains_percentile or 'Not provided'}
- JEE Mains Marks: {inputs.mains_marks or 'Not provided'}
- JEE Mains CRL Rank: {inputs.mains_rank or computed.get('predicted_mains_rank', 'N/A')}
- JEE Advanced Marks: {inputs.adv_marks or 'Not provided'}
- JEE Advanced Rank: {inputs.adv_rank or computed.get('predicted_adv_rank', 'N/A')}
- Category: {cat} ({cat_desc})
- Home State: {inputs.home_state or 'Not specified'}
- Gender: {inputs.gender}
- Academic Interests: {', '.join(inputs.interests) if inputs.interests else 'Undecided'}

## COMPUTED ESTIMATES
- Estimated Mains CRL Rank: {computed.get('predicted_mains_rank', 'N/A')}
- Estimated Mains Percentile: {computed.get('predicted_percentile', 'N/A')}
- Estimated Adv Rank: {computed.get('predicted_adv_rank', 'N/A')}
- Category Rank Relaxation Factor: {relax:.2f}× (vs OPEN candidates)
- Effective Mains Rank for category comparison: {computed.get('effective_cat_rank', 'N/A')}

## YOUR TASK
Write a clear, encouraging rank analysis (300–400 words) covering:

### 1. 📍 Where You Stand
- Explain the percentile/rank in context (out of ~12 lakh candidates)
- Compare to previous year cutoffs briefly

### 2. 🏛 IIT Prospects (via JEE Advanced)
{f"Based on Adv rank {inputs.adv_rank or computed.get('predicted_adv_rank', 'N/A')}, which IITs and branches are realistic?" if (inputs.adv_rank or inputs.adv_marks) else "Adv rank not provided — explain JEE Adv eligibility."}

### 3. 🎓 NIT/IIIT Prospects (via JEE Mains)
Based on CRL rank, list realistic NITs/IIITs by category {cat}.
For category {cat}: mention that closing ranks are ~{relax:.1f}× OPEN ranks, so
more options open up compared to general category candidates.

### 4. 🛤 Strategy
- Should they attempt JEE Advanced?
- Which JoSAA rounds to target?
- Any CSAB/state counselling backup?

### 5. ⚡ Quick Tips
2–3 specific actionable suggestions for the student.

Be warm, specific, and realistic. Use ₹ for INR. Avoid vague advice."""


def _build_ai_tasks_prompt(username: str, subject: str, context: str) -> str:
    return f"""You are an expert JEE study coach who creates highly personalised study plans.

## STUDENT
- Name/Handle: {username}
- Topic to study: {subject}
- Additional context: {context or 'General preparation'}

## YOUR TASK
Generate exactly 5 study tasks for TODAY. Each task must:
1. Be **specific** (not "study calculus" — say "Solve 10 integration by parts problems")
2. Have a **time estimate** in brackets e.g. [45 min]
3. Include a **resource** e.g. (HC Verma Ch.3) or (PYQ 2022)
4. Be **achievable** in one sitting
5. Build **progressively** from easy → hard

Format: one task per line, numbered 1–5.
No preamble, no explanation — just the 5 tasks."""


# ══════════════════════════════════════════════════════════════════════════════
# ADVISOR ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/api/filters")
async def get_filters():
    return {
        "states":          _state_options(),
        "seat_types":      AVAILABLE_SEAT_TYPES,          # what's actually in data
        "all_categories":  ALL_CATEGORIES_FOR_UI,         # full list for UI dropdown
        "genders":         ["male", "female"],
        "quotas":          ordered_quotas(cutoffs_df["Quota"].dropna().unique().tolist()),
        "dataset_note": (
            "This dataset contains OPEN category data only. For OBC-NCL/SC/ST/EWS, "
            "closing ranks are estimated using standard JoSAA relaxation factors."
        ) if AVAILABLE_SEAT_TYPES == ["OPEN"] else None,
    }


@app.get("/api/category-info")
async def category_info(category: str = "OPEN"):
    """Explain the selected category/caste and how it affects JEE counselling."""
    canon = _canonicalize_category(category)
    desc  = CATEGORY_DESCRIPTIONS.get(canon, CATEGORY_DESCRIPTIONS["OPEN"])
    relax = CATEGORY_RANK_RELAXATION.get(canon, 1.0)
    in_data = canon in AVAILABLE_SEAT_TYPES_IN_DATA

    return {
        "category":         canon,
        "display_name":     desc["full_name"],
        "eligibility":      desc["eligibility"],
        "seat_reservation": desc["seat_reservation_pct"],
        "rank_advantage":   desc["rank_advantage"],
        "jee_note":         desc["jee_note"],
        "quota_note":       desc["quota_note"],
        "relaxation_factor": relax,
        "in_dataset":       in_data,
        "dataset_note": (
            f"Your dataset has only OPEN rows. {canon} cutoffs are estimated "
            f"by multiplying OPEN closing ranks by {relax:.2f}×. "
            "For exact figures, add the full JoSAA category-wise cutoff sheet."
        ) if not in_data else None,
        "all_categories": [
            {
                "key":   k,
                "name":  v["full_name"],
                "relax": CATEGORY_RANK_RELAXATION.get(k, 1.0),
            }
            for k, v in CATEGORY_DESCRIPTIONS.items()
        ],
    }


@app.post("/api/rank-predictor")
async def rank_predictor(inp: RankPredictorInput):
    """
    Advanced rank predictor — converts between marks / percentile / rank,
    estimates category-adjusted rank, and returns a branch-chance matrix.
    """
    computed: dict = {}
    warnings: List[str] = []

    # ── Mains chain: marks → percentile → rank ────────────────────────────
    mains_rank_final: Optional[int] = inp.mains_rank

    if inp.mains_marks is not None and inp.mains_percentile is None and inp.mains_rank is None:
        pct = marks_to_percentile_mains(inp.mains_marks)
        computed["predicted_percentile"] = pct
        computed["predicted_mains_rank"] = percentile_to_rank(pct)
        mains_rank_final = computed["predicted_mains_rank"]

    elif inp.mains_percentile is not None and inp.mains_rank is None:
        computed["predicted_mains_rank"] = percentile_to_rank(inp.mains_percentile)
        mains_rank_final = computed["predicted_mains_rank"]

    elif inp.mains_rank is not None and inp.mains_percentile is None:
        computed["predicted_percentile"] = rank_to_percentile(inp.mains_rank)

    if inp.mains_marks is not None and "predicted_percentile" not in computed:
        computed["predicted_percentile"] = marks_to_percentile_mains(inp.mains_marks)

    # ── Advanced chain: marks → rank ──────────────────────────────────────
    adv_rank_final: Optional[int] = inp.adv_rank
    if inp.adv_marks is not None and inp.adv_rank is None:
        computed["predicted_adv_rank"] = marks_to_rank_advanced(inp.adv_marks)
        adv_rank_final = computed["predicted_adv_rank"]

    # ── Category-adjusted effective rank ─────────────────────────────────
    canon_cat = _canonicalize_category(inp.category)
    relax     = CATEGORY_RANK_RELAXATION.get(canon_cat, 1.0)
    if mains_rank_final:
        # Effective rank = the OPEN rank threshold that corresponds to the
        # student's position in the category queue. Conceptually, an OBC
        # student ranked 2700 in OBC list competes for seats where OPEN
        # closing rank may be ~2000 (2700 / 1.35).
        computed["effective_cat_rank"]     = mains_rank_final
        computed["equivalent_open_rank"]   = int(round(mains_rank_final / relax))
        computed["category_relaxation_pct"] = round((relax - 1.0) * 100, 1)

    # ── Branch-chance matrix ──────────────────────────────────────────────
    branch_chances = []
    if mains_rank_final:
        branch_chances = predict_branch_chances(mains_rank_final, inp.home_state, inp.category)

    # ── AI narrative ──────────────────────────────────────────────────────
    ai_narrative = ""
    if groq_client and (mains_rank_final or adv_rank_final):
        try:
            prompt = _build_rank_predictor_prompt(inp, computed)
            resp   = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=900,
                temperature=0.4,
            )
            ai_narrative = resp.choices[0].message.content
        except Exception as e:
            ai_narrative = f"AI narrative unavailable: {e}"

    # ── Category note ─────────────────────────────────────────────────────
    cat_in_data = canon_cat in AVAILABLE_SEAT_TYPES_IN_DATA
    if not cat_in_data and canon_cat != "OPEN":
        warnings.append(
            f"Your dataset contains only OPEN category data. "
            f"{canon_cat} closing ranks are estimated by applying a {relax:.2f}× "
            f"relaxation factor to OPEN ranks. Add the full JoSAA "
            f"category-wise sheet for exact figures."
        )

    return {
        "inputs":          inp.dict(),
        "computed":        computed,
        "mains_rank":      mains_rank_final,
        "adv_rank":        adv_rank_final,
        "category":        canon_cat,
        "category_info":   CATEGORY_DESCRIPTIONS.get(canon_cat, {}),
        "relaxation":      relax,
        "branch_chances":  branch_chances,
        "ai_narrative":    ai_narrative,
        "warnings":        warnings,
    }


@app.post("/api/college-compare")
async def college_compare(req: CollegeCompareRequest):
    """Side-by-side AI comparison of 2–4 colleges."""
    if len(req.college_names) < 2:
        raise HTTPException(400, "Provide at least 2 college names to compare.")
    if len(req.college_names) > 4:
        raise HTTPException(400, "Maximum 4 colleges can be compared at once.")

    colleges_data = []
    for name in req.college_names:
        info = college_info.get(name.strip(), {})
        colleges_data.append({
            "name":      name.strip(),
            "location":  info.get("location", "Unknown"),
            "website":   info.get("website", ""),
            "inst_type": classify_institute(name),
            "category":  info.get("category", "") or classify_institute(name),
        })

    ai_comparison = "Set GROQ_API_KEY for AI-powered college comparison."
    if groq_client:
        try:
            prompt = _build_college_compare_prompt(
                colleges_data, req.branch, req.student_rank, req.category
            )
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=0.4,
            )
            ai_comparison = resp.choices[0].message.content
        except Exception as e:
            ai_comparison = f"AI comparison error: {e}"

    # Also pull cutoff data for each college
    cutoff_summary = {}
    for col in colleges_data:
        prog_df = cutoffs_df[cutoffs_df["Institute"] == col["name"]]
        cutoff_summary[col["name"]] = [
            {
                "program":      str(r["Academic Program Name"]),
                "quota":        str(r["Quota"]),
                "seat_type":    str(r["Seat Type"]),
                "closing_rank": int(r["Closing Rank"]),
            }
            for _, r in prog_df.head(5).iterrows()
        ]

    return {
        "colleges":       colleges_data,
        "cutoff_summary": cutoff_summary,
        "ai_comparison":  ai_comparison,
        "category":       req.category,
        "branch":         req.branch,
    }


@app.post("/api/get-options")
async def get_options(profile: StudentProfile):
    if not profile.jee_mains_rank and not profile.jee_adv_rank:
        raise HTTPException(400, "Provide at least one rank.")

    filtered, is_exact = filter_options(profile)
    canon_cat = _canonicalize_category(str(profile.seat_type or profile.category or "OPEN"))
    relax     = CATEGORY_RANK_RELAXATION.get(canon_cat, 1.0)

    category_note = ""
    if not is_exact and canon_cat != "OPEN":
        category_note = (
            f"Your dataset contains only OPEN category data. "
            f"Closing ranks shown for {canon_cat} are estimated by multiplying "
            f"OPEN closing ranks by {relax:.2f}×. These are approximations — "
            f"actual JoSAA cutoffs may differ slightly."
        )

    if filtered.empty:
        return {
            "options":      [],
            "groups":       {"IIT": [], "NIT": [], "IIIT": [], "Other": []},
            "message":      "No matching options found.",
            "category_note": category_note,
        }

    records = []
    seen    = set()
    INST_ORDER = {"IIT": 0, "NIT": 1, "IIIT": 2, "Other": 3}

    for _, row in filtered.iterrows():
        inst   = str(row["Institute"])
        branch = str(row["Academic Program Name"])
        key    = (inst, branch, str(row["Quota"]))
        if key in seen:
            continue
        seen.add(key)

        inst_type    = str(row.get("_inst_type", classify_institute(inst)))
        student_rank = int(row["_student_rank"])
        opening_rank = float(row["Opening Rank"])
        closing_rank = float(row["Closing Rank"])

        prob  = compute_probability(student_rank, opening_rank, closing_rank, inst)
        label = chance_label(prob)
        if label == "Good" and student_rank >= closing_rank * 0.97:
            label = "Moderate"
            prob  = min(prob, _MODERATE_MIN_PROB + 10)

        pref_score   = score_branch_for_interests(branch, profile.interests)
        in_safe_zone = student_rank < opening_rank

        college_meta  = college_info.get(inst, {})
        college_state = _find_state_from_location(college_meta.get("location", ""))
        state_priority = _state_distance_rank(profile.home_state or "", college_state)

        records.append({
            "institute":           inst,
            "program":             branch,
            "quota":               str(row["Quota"]),
            "seat_type":           str(row["Seat Type"]),
            "gender":              str(row["Gender"]),
            "opening_rank":        int(opening_rank),
            "closing_rank":        int(closing_rank),
            "student_rank_used":   student_rank,
            "rank_source":         "JEE Advanced" if inst_type == "IIT" else "JEE Mains",
            "website":             college_meta.get("website", "") or str(row.get("_website", "")),
            "location":            college_meta.get("location", "") or str(row.get("_location", "")),
            "college_state":       college_state or "",
            "state_priority":      state_priority,
            "college_category":    str(row.get("_category", inst_type)),
            "inst_type":           inst_type,
            "is_iit":              inst_type == "IIT",
            "preference_score":    pref_score,
            "chance":              label,
            "chance_probability":  prob,
            "in_safe_zone":        in_safe_zone,
            "category_estimated":  not is_exact,
        })

    if profile.nearest_only:
        records = [r for r in records if r.get("state_priority", 3) in (0, 1)]

    records.sort(key=lambda x: (
        x.get("state_priority", 3),
        INST_ORDER.get(x["inst_type"], 4),
        0 if x["in_safe_zone"] else 1,
        -x["preference_score"],
        -x["chance_probability"],
        x["closing_rank"],
    ))

    groups: Dict[str, List[dict]] = {"IIT": [], "NIT": [], "IIIT": [], "Other": []}
    for r in records:
        groups[r["inst_type"]].append(r)

    return {
        "options":         records,
        "groups":          groups,
        "total":           len(records),
        "rank_info": {
            "mains_used": bool(profile.jee_mains_rank),
            "adv_used":   bool(profile.jee_adv_rank),
            "mains_rank": profile.jee_mains_rank,
            "adv_rank":   profile.jee_adv_rank,
        },
        "category":        canon_cat,
        "category_note":   category_note,
        "is_exact_category_data": is_exact,
        "relaxation_factor": relax if not is_exact else 1.0,
    }


@app.post("/api/college-detail")
async def college_detail(req: CollegeDetailRequest):
    name     = req.college_name.strip()
    info     = college_info.get(name, {})
    website  = info.get("website", "") or _fallback_website(name)
    location = info.get("location", "")
    category = info.get("category", "") or classify_institute(name)

    programs_df = cutoffs_df[cutoffs_df["Institute"] == name]
    programs    = [
        {
            "program":      str(r["Academic Program Name"]),
            "quota":        str(r["Quota"]),
            "seat_type":    str(r["Seat Type"]),
            "gender":       str(r["Gender"]),
            "opening_rank": int(r["Opening Rank"]),
            "closing_rank": int(r["Closing Rank"]),
        }
        for _, r in programs_df.iterrows()
    ]

    scrape_data = None
    if req.use_scraper:
        loop = asyncio.get_event_loop()
        scrape_data = await loop.run_in_executor(
            _executor, scrape_college_info, name, req.branch
        )

    insights = await _get_college_insights(name, category, location, programs, scrape_data)

    return {
        "name":        name,
        "website":     website,
        "location":    location,
        "category":    category,
        "inst_type":   classify_institute(name),
        "programs":    programs,
        "insights":    insights,
        "scrape_data": scrape_data if scrape_data and scrape_data.get("raw_text") else None,
    }


def _fallback_website(name: str) -> str:
    KNOWN = {
        "Indian Institute of Technology Bombay":    "https://www.iitb.ac.in",
        "Indian Institute of Technology Delhi":     "https://www.iitd.ac.in",
        "Indian Institute of Technology Madras":    "https://www.iitm.ac.in",
        "Indian Institute of Technology Kanpur":    "https://www.iitk.ac.in",
        "Indian Institute of Technology Kharagpur": "https://www.iitkgp.ac.in",
        "Indian Institute of Technology Roorkee":   "https://www.iitr.ac.in",
        "Indian Institute of Technology Guwahati":  "https://www.iitg.ac.in",
        "Indian Institute of Technology Hyderabad": "https://www.iith.ac.in",
        "National Institute of Technology, Tiruchirappalli": "https://www.nitt.edu",
        "National Institute of Technology Warangal":         "https://www.nitw.ac.in",
        "National Institute of Technology Karnataka, Surathkal": "https://www.nitk.ac.in",
        "Motilal Nehru National Institute of Technology Allahabad": "https://www.mnnit.ac.in",
        "Visvesvaraya National Institute of Technology, Nagpur": "https://www.vnit.ac.in",
    }
    for k, v in KNOWN.items():
        if k.lower() in name.lower():
            return v
    if "iit" in name.lower():
        city = name.strip().split()[-1].lower()
        return f"https://www.iit{city}.ac.in"
    return ""


async def _get_college_insights(name: str, category: str, location: str,
                                programs: List[dict],
                                scrape_data: Optional[dict] = None,
                                student_category: str = "OPEN") -> str:
    cached = await asyncio.get_event_loop().run_in_executor(
        _executor, db_get_college_insights, name
    )
    if cached and not scrape_data:
        return cached

    if not groq_client:
        return _static_insights(name, category)

    try:
        prompt = _build_college_insight_prompt(
            name, category, location, programs, scrape_data, student_category
        )
        resp   = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3,
        )
        insights = resp.choices[0].message.content
        await asyncio.get_event_loop().run_in_executor(
            _executor, db_set_college_insights, name, insights
        )
        return insights
    except Exception as e:
        print(f"[GROQ college-insight] {e}")
        return _static_insights(name, category)


def _static_insights(name: str, category: str) -> str:
    if "IIT" in name:
        return (
            f"{name} is a premier IIT. Known for world-class research, "
            "excellent placements (avg ₹15–25 LPA, top >₹1 Cr in CS). "
            "Set GROQ_API_KEY for detailed AI insights."
        )
    if "NIT" in name:
        return (
            f"{name} is an NIT offering quality technical education. "
            "Typical placements: avg ₹8–15 LPA. "
            "Set GROQ_API_KEY for detailed AI insights."
        )
    return f"Set GROQ_API_KEY for detailed AI-powered insights about {name}."


# ── College Enquiry ────────────────────────────────────────────────────────────

@app.post("/api/college-enquiry")
async def college_enquiry(req: CollegeEnquiryIn):
    loop = asyncio.get_event_loop()
    scrape_data = await loop.run_in_executor(
        _executor, scrape_college_info, req.college_name, req.branch
    )

    if not req.question.strip():
        return {
            "college": req.college_name,
            "branch":  req.branch,
            "scrape":  scrape_data,
            "answer":  scrape_data.get("raw_text", "No data found.")[:2000],
            "source":  "perplexity_scrape",
        }

    if not groq_client:
        return {
            "college": req.college_name,
            "answer":  "Set GROQ_API_KEY to enable AI-powered answers.",
            "source":  "static",
        }

    try:
        prompt = _build_enquiry_prompt(
            req.college_name, req.branch, req.question, scrape_data
        )
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.4,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"AI error: {e}"

    return {
        "college":  req.college_name,
        "branch":   req.branch,
        "question": req.question,
        "answer":   answer,
        "scrape":   scrape_data,
        "source":   "groq+perplexity",
    }


# ── Chat (advisor) ────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(msg: ChatMessage):
    if not groq_client and not gemini_client:
        return {
            "reply": "⚠️ No AI client is configured. Set GROQ_API_KEY or GOOGLE_GEMINI_API_KEY in your environment."
        }

    history = msg.history or []

    options_summary  = ""
    category_note    = ""
    college_list     = msg.college_list or []

    if msg.student_profile:
        try:
            filtered, is_exact = filter_options(msg.student_profile)
            if not filtered.empty:
                lines = []
                candidate_list = []
                for _, r in filtered.head(15).iterrows():
                    lines.append(
                        f"- {r['Institute']} | {r['Academic Program Name']} | "
                        f"Quota: {r['Quota']} | Closing: {r['Closing Rank']}"
                    )
                    candidate_list.append({
                        "institute": str(r["Institute"]),
                        "program": str(r["Academic Program Name"]),
                        "quota": str(r["Quota"]),
                        "seat_type": str(r["Seat Type"]),
                        "opening_rank": int(r["Opening Rank"]),
                        "closing_rank": int(r["Closing Rank"]),
                        "location": str(r.get("_location", "") or ""),
                        "chance": "",
                    })
                options_summary = "\n".join(lines)
                if not college_list:
                    college_list = candidate_list
            if not is_exact:
                canon_cat = _canonicalize_category(
                    str(msg.student_profile.seat_type or msg.student_profile.category or "OPEN")
                )
                relax = CATEGORY_RANK_RELAXATION.get(canon_cat, 1.0)
                category_note = (
                    f"Dataset has OPEN-only data. {canon_cat} cutoffs estimated "
                    f"at {relax:.2f}× OPEN closing ranks. Treat as approximate."
                )
        except Exception:
            pass
    elif college_list:
        lines = []
        for college in college_list[:15]:
            lines.append(
                f"- {college.get('institute', 'Unknown')} | {college.get('program', '')} | "
                f"Quota: {college.get('quota', '')} | Closing: {college.get('closing_rank', '')}"
            )
        options_summary = "\n".join(lines)

    perplexity_data = None
    college_name    = msg.college_name or None
    college_branch  = msg.college_branch or ""
    if not college_name and msg.use_perplexity and college_list:
        first = college_list[0]
        college_name = first.get("institute") or first.get("college")
        college_branch = college_branch or first.get("program", "")

    if not college_name and msg.message:
        college_name = _extract_college_name_from_message(msg.message)
    if college_name and msg.use_perplexity and not gemini_client:
        loop = asyncio.get_event_loop()
        try:
            perplexity_data = await loop.run_in_executor(
                _executor, scrape_college_info, college_name, college_branch
            )
        except Exception as e:
            print(f"[CHAT] Perplexity scrape failed: {e}")

    system   = _build_counselor_system(
        msg.student_profile,
        options_summary,
        category_note,
        perplexity_data,
        college_list=college_list,
    )
    messages = [{"role": "system", "content": system}]
    for h in history[-12:]:
        if h.get("role") in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": msg.message})

    try:
        if gemini_client:
            gemini_prompt = system + "\n\n" + msg.message
            response = gemini_client.interactions.create(
                model=GOOGLE_GEMINI_MODEL,
                input=gemini_prompt,
            )
            reply = getattr(response, "output_text", None) or str(response)
        else:
            resp  = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=1200,
                temperature=0.5,
            )
            reply = resp.choices[0].message.content
    except Exception as e:
        reply = f"⚠️ Error from AI: {e}"

    updated_history = history + [
        {"role": "user",      "content": msg.message},
        {"role": "assistant", "content": reply},
    ]

    return {"reply": reply, "history": updated_history}


@app.get("/api/health")
async def health():
    return {
        "status":              "ok",
        "cutoffs_rows":        len(cutoffs_df),
        "colleges":            len(colleges_df),
        "seat_types_in_data":  AVAILABLE_SEAT_TYPES,
        "all_categories_ui":   ALL_CATEGORIES_FOR_UI,
        "ai_enabled":          groq_client is not None or gemini_client is not None,
        "google_gemini_enabled": gemini_client is not None,
        "mega_enabled":        _mega.m is not None,
        "version":             "7.0.0",
        "storage_mode":        "tidb+mega",
        "new_endpoints": [
            "GET  /api/category-info?category=OBC-NCL",
            "POST /api/rank-predictor",
            "POST /api/college-compare",
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# MEDIA UPLOAD — MEGA cloud (binary only)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/community/api/upload")
async def upload_media(file: UploadFile = File(...)):
    ct = _resolve_content_type(file.content_type or "", file.filename or "")
    if ct not in ALLOWED_ALL_TYPES:
        raise HTTPException(
            400,
            f"Unsupported file type '{file.content_type}' (resolved: '{ct}'). "
            "Allowed: images, videos, PDF, Word, Excel, PowerPoint."
        )

    data = await file.read(MAX_FILE_SIZE + 1)
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large (max 50 MB)")

    ext      = Path(file.filename or "upload").suffix.lower() or ".bin"
    filename = f"{uuid.uuid4().hex}{ext}"

    staging_path = str(STAGING_DIR / filename)
    with open(staging_path, "wb") as f:
        f.write(data)

    loop = asyncio.get_event_loop()
    try:
        public_url = await loop.run_in_executor(
            _executor, _mega.upload_get_link, staging_path, filename
        )
    except RuntimeError as e:
        raise HTTPException(502, f"Cloud upload failed: {e}")

    if ct in ALLOWED_IMAGE_TYPES:
        media_type = "image"
    elif ct in ALLOWED_VIDEO_TYPES:
        media_type = "video"
    else:
        media_type = "document"

    return {"url": public_url, "media_type": media_type, "filename": filename, "cloud": True}


@app.delete("/community/api/media/{filename}")
async def delete_media(filename: str, author: str = Query(...)):
    filename = Path(filename).name
    loop = asyncio.get_event_loop()

    def _check_owner() -> bool:
        from database import execute
        r = execute(
            "SELECT id FROM posts WHERE media_url LIKE %s AND author=%s LIMIT 1",
            (f"%{filename}%", author), fetch="one"
        )
        if r: return True
        r = execute(
            "SELECT id FROM comments WHERE media_url LIKE %s AND author=%s LIMIT 1",
            (f"%{filename}%", author), fetch="one"
        )
        if r: return True
        r = execute(
            "SELECT id FROM group_messages WHERE media_url LIKE %s AND author=%s LIMIT 1",
            (f"%{filename}%", author), fetch="one"
        )
        return bool(r)

    is_owner = await loop.run_in_executor(_executor, _check_owner)
    if not is_owner:
        raise HTTPException(403, "Not authorized to delete this file")

    _executor.submit(_mega.delete_file, filename)
    return {"deleted": filename}


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKETS
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/feed")
async def ws_feed(ws: WebSocket):
    await ws_manager.connect(ws, "feed")
    try:
        while True:
            await asyncio.sleep(25)
            await ws.send_json({"type": "ping"})
    except (WebSocketDisconnect, Exception):
        await ws_manager.disconnect(ws, "feed")


@app.websocket("/ws/group/{group_id}")
async def ws_group(ws: WebSocket, group_id: str):
    channel = f"group:{group_id}"
    await ws_manager.connect(ws, channel)
    try:
        while True:
            await asyncio.sleep(25)
            await ws.send_json({"type": "ping"})
    except (WebSocketDisconnect, Exception):
        await ws_manager.disconnect(ws, channel)


# ══════════════════════════════════════════════════════════════════════════════
# COMMUNITY — POSTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/community/api/posts")
async def c_get_posts(page: int = 0):
    loop  = asyncio.get_event_loop()
    posts = await loop.run_in_executor(_executor, db_get_posts, page, POSTS_PER_PAGE)
    total = await loop.run_in_executor(_executor, db_count_posts)
    return {
        "posts":    posts,
        "has_more": (page + 1) * POSTS_PER_PAGE < total,
    }


@app.post("/community/api/posts")
async def c_create_post(body: PostIn):
    if not body.content.strip() and not body.media_url:
        raise HTTPException(400, "Empty post")
    post = {
        "id":           str(uuid.uuid4()),
        "author":       body.author,
        "avatar_color": body.avatar_color,
        "content":      body.content.strip()[:500],
        "media_url":    body.media_url,
        "media_type":   body.media_type,
        "timestamp":    _now(),
        "reactions":    {},
        "comment_count": 0,
    }
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, db_create_post, post)
    await ws_manager.broadcast("feed", {"type": "new_post", "post": post})
    return {"post": post}


@app.delete("/community/api/posts/{post_id}")
async def c_delete_post(post_id: str, author: str = Query(...)):
    loop = asyncio.get_event_loop()
    try:
        row = await loop.run_in_executor(_executor, db_delete_post, post_id, author)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    if not row:
        raise HTTPException(404, "Post not found")
    if row.get("media_url"):
        _executor.submit(_mega.delete_file, Path(row["media_url"]).name)
    await ws_manager.broadcast("feed", {"type": "post_deleted", "post_id": post_id})
    return {"deleted": post_id}


@app.post("/community/api/posts/{post_id}/react")
async def c_react_post(post_id: str, body: ReactIn):
    loop = asyncio.get_event_loop()
    p    = await loop.run_in_executor(_executor, db_get_post, post_id)
    if not p:
        raise HTTPException(404, "Post not found")
    reactions = await loop.run_in_executor(_executor, db_toggle_reaction, post_id, body.emoji, body.author)
    await ws_manager.broadcast("feed", {
        "type": "reaction_update", "post_id": post_id, "reactions": reactions,
    })
    return {"reactions": reactions}


@app.get("/community/api/posts/{post_id}/comments")
async def c_get_comments(post_id: str):
    loop     = asyncio.get_event_loop()
    comments = await loop.run_in_executor(_executor, db_get_comments, post_id)
    return {"comments": comments}


@app.post("/community/api/posts/{post_id}/comments")
async def c_add_comment(post_id: str, body: CommentIn):
    if not body.content.strip() and not body.media_url:
        raise HTTPException(400, "Empty comment")
    comment = {
        "id":           str(uuid.uuid4()),
        "post_id":      post_id,
        "author":       body.author,
        "avatar_color": body.avatar_color,
        "content":      body.content.strip()[:300],
        "media_url":    body.media_url,
        "media_type":   body.media_type,
        "timestamp":    _now(),
    }
    loop    = asyncio.get_event_loop()
    comment = await loop.run_in_executor(_executor, db_add_comment, comment)
    await ws_manager.broadcast("feed", {
        "type":          "comment_added",
        "post_id":       post_id,
        "comment_count": comment.get("comment_count", 0),
    })
    return {"comment": comment}


@app.delete("/community/api/posts/{post_id}/comments/{comment_id}")
async def c_delete_comment(post_id: str, comment_id: str, author: str = Query(...)):
    loop = asyncio.get_event_loop()
    try:
        row = await loop.run_in_executor(_executor, db_delete_comment, post_id, comment_id, author)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    if not row:
        raise HTTPException(404, "Comment not found")
    if row.get("media_url"):
        _executor.submit(_mega.delete_file, Path(row["media_url"]).name)
    return {"deleted": comment_id}


# ══════════════════════════════════════════════════════════════════════════════
# COMMUNITY — GROUPS
# ══════════════════════════════════════════════════════════════════════════════

def _unique_code() -> str:
    from database import execute
    for _ in range(10):
        code = _gen_code()
        if not execute("SELECT 1 FROM `groups` WHERE code=%s", (code,), fetch="one"):
            return code
    return _gen_code()


@app.get("/community/api/groups/search")
async def c_search_groups(q: str = ""):
    loop = asyncio.get_event_loop()
    groups = await loop.run_in_executor(_executor, db_get_groups_public, q.strip())
    return {"groups": groups}


@app.get("/community/api/groups/mine")
async def c_my_groups(user: str = Query(...)):
    loop   = asyncio.get_event_loop()
    groups = await loop.run_in_executor(_executor, db_get_my_groups, user)
    return {"groups": groups}


@app.post("/community/api/groups")
async def c_create_group(body: GroupCreateIn):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Group name required")

    expiry_months = max(1, min(body.expiry_months, 6))
    expiry = (datetime.now(timezone.utc) + timedelta(days=30 * expiry_months)).strftime("%Y-%m-%d %H:%M:%S")

    loop = asyncio.get_event_loop()
    code = await loop.run_in_executor(_executor, _unique_code)

    group = {
        "id":            str(uuid.uuid4()),
        "name":          name,
        "subject":       body.subject.strip(),
        "code":          code,
        "group_type":    body.group_type,
        "expiry_months": expiry_months,
        "expiry_date":   expiry,
        "created_at":    _now(),
        "created_by":    body.created_by,
    }

    await loop.run_in_executor(
        _executor, db_create_group, group, body.created_by, body.avatar_color
    )
    return {"group": group, "code": code}


@app.post("/community/api/groups/join")
async def c_join_group(body: GroupJoinIn):
    code   = body.code.strip().upper()
    loop   = asyncio.get_event_loop()
    target = await loop.run_in_executor(_executor, db_get_group_by_code, code)
    if not target:
        raise HTTPException(404, "Invalid invite code")

    group_id  = target["id"]
    is_member = await loop.run_in_executor(_executor, db_is_member, group_id, body.username)
    if is_member:
        safe = {k: v for k, v in target.items() if k != "code"}
        return {"group": safe, "already_member": True}

    if target.get("group_type", "public") == "private":
        has_req = await loop.run_in_executor(_executor, db_has_request, group_id, body.username)
        if has_req:
            return {"request_pending": True, "message": "Join request already pending."}
        await loop.run_in_executor(_executor, db_add_request, group_id, body.username, body.avatar_color)
        payload = {
            "type": "join_request",
            "group_id": group_id,
            "group_name": target.get("name"),
            "username": body.username,
            "avatar_color": body.avatar_color,
            "created_by": target.get("created_by"),
        }
        await ws_manager.broadcast(f"group:{group_id}", payload)
        await ws_manager.broadcast("feed", payload)
        return {"request_pending": True, "message": "Join request sent to the group owner."}

    await loop.run_in_executor(_executor, db_add_member, group_id, body.username, body.avatar_color)
    safe = {k: v for k, v in target.items() if k != "code"}
    return {"group": safe, "already_member": False}


@app.get("/community/api/groups/{group_id}/requests")
async def c_get_group_requests(group_id: str, owner: str = Query(...)):
    loop   = asyncio.get_event_loop()
    target = await loop.run_in_executor(_executor, db_get_group, group_id)
    if not target:
        raise HTTPException(404, "Group not found")
    if target.get("created_by") != owner:
        raise HTTPException(403, "Only the group owner can view requests")
    reqs = await loop.run_in_executor(_executor, db_get_requests, group_id)
    return {"requests": reqs, "group_id": group_id}


@app.post("/community/api/groups/{group_id}/requests/{username}/approve")
async def c_approve_group_request(group_id: str, username: str, owner: str = Query(...)):
    loop   = asyncio.get_event_loop()
    target = await loop.run_in_executor(_executor, db_get_group, group_id)
    if not target or target.get("created_by") != owner:
        raise HTTPException(403, "Not authorized")
    req = next((r for r in await loop.run_in_executor(_executor, db_get_requests, group_id)
                if r["username"] == username), None)
    if not req:
        raise HTTPException(404, "Request not found")
    await loop.run_in_executor(_executor, db_add_member, group_id, username, req.get("avatar_color", "#5865F2"))
    await loop.run_in_executor(_executor, db_remove_request, group_id, username)
    payload = {
        "type": "request_approved",
        "group_id": group_id,
        "group_name": target.get("name"),
        "username": username,
        "created_by": target.get("created_by"),
    }
    await ws_manager.broadcast(f"group:{group_id}", payload)
    await ws_manager.broadcast("feed", payload)
    return {"approved": True, "member": username}


@app.post("/community/api/groups/{group_id}/requests/{username}/reject")
async def c_reject_group_request(group_id: str, username: str, owner: str = Query(...)):
    loop   = asyncio.get_event_loop()
    target = await loop.run_in_executor(_executor, db_get_group, group_id)
    if not target or target.get("created_by") != owner:
        raise HTTPException(403, "Not authorized")
    await loop.run_in_executor(_executor, db_remove_request, group_id, username)
    payload = {
        "type": "request_rejected",
        "group_id": group_id,
        "group_name": target.get("name"),
        "username": username,
        "created_by": target.get("created_by"),
    }
    await ws_manager.broadcast(f"group:{group_id}", payload)
    await ws_manager.broadcast("feed", payload)
    return {"rejected": username}


@app.delete("/community/api/groups/{group_id}/members/{member_username}")
async def c_remove_group_member(group_id: str, member_username: str, owner: str = Query(...)):
    loop   = asyncio.get_event_loop()
    target = await loop.run_in_executor(_executor, db_get_group, group_id)
    if not target or target.get("created_by") != owner:
        raise HTTPException(403, "Not authorized")
    if member_username == owner:
        raise HTTPException(400, "Owner cannot remove themselves")
    await loop.run_in_executor(_executor, db_remove_member, group_id, member_username)
    return {"removed": member_username}


@app.post("/community/api/groups/{group_id}/leave")
async def c_leave_group(group_id: str, body: GroupLeaveIn):
    loop   = asyncio.get_event_loop()
    target = await loop.run_in_executor(_executor, db_get_group, group_id)
    if not target:
        raise HTTPException(404, "Group not found")
    if target.get("created_by") == body.username:
        raise HTTPException(400, "Owner must delete the group instead of leaving")
    await loop.run_in_executor(_executor, db_remove_member, group_id, body.username)
    return {"left": body.username}


@app.delete("/community/api/groups/{group_id}")
async def c_delete_group(group_id: str, owner: str = Query(...)):
    loop = asyncio.get_event_loop()
    ok   = await loop.run_in_executor(_executor, db_delete_group, group_id, owner)
    if not ok:
        raise HTTPException(403, "Not authorized or group not found")
    return {"deleted": group_id}


@app.get("/community/api/groups/{group_id}/messages")
async def c_get_messages(group_id: str, after: str = ""):
    loop = asyncio.get_event_loop()
    g    = await loop.run_in_executor(_executor, db_get_group, group_id)
    if not g:
        raise HTTPException(404, "Group not found")
    msgs = await loop.run_in_executor(_executor, db_get_messages, group_id, after)
    return {"messages": msgs, "group_id": group_id}


@app.post("/community/api/groups/{group_id}/messages")
async def c_send_message(group_id: str, body: MsgIn):
    loop = asyncio.get_event_loop()
    g    = await loop.run_in_executor(_executor, db_get_group, group_id)
    if not g:
        raise HTTPException(404, "Group not found")
    if not body.content.strip() and not body.media_url:
        raise HTTPException(400, "Empty message")

    msg = {
        "id":           str(uuid.uuid4()),
        "group_id":     group_id,
        "author":       body.author,
        "avatar_color": body.avatar_color,
        "content":      body.content.strip()[:1000],
        "media_url":    body.media_url,
        "media_type":   body.media_type,
        "timestamp":    _now(),
    }
    await loop.run_in_executor(_executor, db_send_message, msg)
    await ws_manager.broadcast(f"group:{group_id}", {
        "type": "new_message", "message": msg, "group_id": group_id,
    })
    return {"message": msg}


@app.delete("/community/api/groups/{group_id}/messages/{msg_id}")
async def c_delete_group_msg(group_id: str, msg_id: str, author: str = Query(...)):
    loop = asyncio.get_event_loop()
    try:
        row = await loop.run_in_executor(_executor, db_delete_message, group_id, msg_id, author)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    if not row:
        raise HTTPException(404, "Message not found")
    if row.get("media_url"):
        _executor.submit(_mega.delete_file, Path(row["media_url"]).name)
    await ws_manager.broadcast(f"group:{group_id}", {
        "type": "message_deleted", "msg_id": msg_id, "group_id": group_id,
    })
    return {"deleted": msg_id}


@app.get("/community/api/groups/{group_id}/members")
async def c_get_members(group_id: str):
    loop = asyncio.get_event_loop()
    mbrs = await loop.run_in_executor(_executor, db_get_members, group_id)
    return {"members": mbrs, "group_id": group_id}


# ══════════════════════════════════════════════════════════════════════════════
# AI STUDY TASKS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/community/api/ai/tasks")
async def c_ai_tasks(body: AiTaskIn):
    if not groq_client:
        return {"tasks": [{"id": 1, "text": "Set GROQ_API_KEY to enable AI tasks", "done": False}]}

    prompt = _build_ai_tasks_prompt(body.username, body.subject, body.context)
    try:
        resp  = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
        )
        text  = resp.choices[0].message.content.strip()
        tasks = []
        for i, line in enumerate([l.strip() for l in text.split("\n") if l.strip()][:5], 1):
            cleaned = re.sub(r"^[\d]+[.)]\s*", "", line).strip()
            if cleaned:
                tasks.append({"id": i, "text": cleaned, "done": False})
        return {"tasks": tasks or [{"id": 1, "text": text[:100], "done": False}]}
    except Exception as e:
        return {"tasks": [{"id": 1, "text": f"AI error: {str(e)[:80]}", "done": False}]}
