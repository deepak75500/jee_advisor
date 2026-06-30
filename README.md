# JEE College Advisor + StudyCord Community
### Production v7.0 — Cloud-Native AI-Powered College Counselling Platform

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Tech Stack](#tech-stack)
4. [Core Algorithm: College Recommendation Engine](#core-algorithm-college-recommendation-engine)
5. [Rank Predictor Algorithm](#rank-predictor-algorithm)
6. [Category / Caste Reservation System](#category--caste-reservation-system)
7. [Quota & Home-State Logic](#quota--home-state-logic)
8. [Probability Scoring Engine](#probability-scoring-engine)
9. [Interest-to-Branch Matching](#interest-to-branch-matching)
10. [AI Counselor (Groq LLM)](#ai-counselor-groq-llm)
11. [StudyCord Community Platform](#studycord-community-platform)
12. [Perplexity Scraper](#perplexity-scraper)
13. [Database Schema (TiDB)](#database-schema-tidb)
14. [MEGA Media Store](#mega-media-store)
15. [API Reference](#api-reference)
16. [Environment Variables](#environment-variables)
17. [Running Locally](#running-locally)
18. [Data Files](#data-files)

---

## Project Overview

The **JEE College Advisor** is a full-stack web platform that helps JEE aspirants make data-driven college and branch choices. It combines:

- **Deterministic filtering** using JoSAA 2025 cutoff data (128 colleges, all seat types)
- **Statistical probability scoring** using a sigmoid-based admission chance model
- **AI-powered counselling** via Groq (LLaMA-3.3-70B) with prompt engineering
- **Live data enrichment** via a Selenium scraper querying Perplexity.ai
- **Community platform (StudyCord)** — Discord-like study groups, posts, reactions, and real-time chat over WebSocket
- **Cloud persistence** via TiDB Cloud (MySQL-compatible) for all structured data and MEGA for binary media

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Browser                           │
│   index.html (Jinja2 template, ~132 KB, vanilla JS + CSS)       │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP / WebSocket
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FastAPI  (main.py  v7.0)                        │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  Advisor Routes  │  │  Community Routes│  │  WS Manager  │  │
│  │  /api/get-options│  │  /api/posts/*    │  │  (asyncio)   │  │
│  │  /api/chat       │  │  /api/groups/*   │  └──────────────┘  │
│  │  /api/rank-pred. │  │  /api/messages/* │                    │
│  │  /api/college-*  │  │  /api/media/*    │                    │
│  └────────┬─────────┘  └────────┬─────────┘                    │
│           │                     │                               │
│   ┌───────┴──────────────────────┴──────────────────────────┐   │
│   │              Core Engine (main.py internals)            │   │
│   │  filter_options()  compute_probability()  _interpolate()│   │
│   │  predict_branch_chances()  score_branch_for_interests() │   │
│   │  get_applicable_quotas()   resolve_category_seat_types()│   │
│   └──────────────────────────────────────────────────────── ┘   │
└──────────┬────────────────┬──────────────┬──────────────────────┘
           │                │              │
           ▼                ▼              ▼
  ┌────────────────┐ ┌──────────────┐ ┌──────────────────┐
  │   TiDB Cloud   │ │  Groq API    │ │   Selenium /     │
  │  (database.py) │ │  LLaMA-3.3   │ │ Perplexity.ai    │
  │  10 tables     │ │  -70B-vers.  │ │ (p_scraper.py)   │
  └────────────────┘ └──────────────┘ └──────────────────┘
           │
           ▼
  ┌────────────────┐
  │  MEGA Cloud    │
  │  (media only)  │
  │  images/videos │
  │  /documents    │
  └────────────────┘
```

**Data Flow for a College Recommendation Request:**

```
User inputs rank + state + category + interests
        │
        ▼
filter_options(profile)
  ├── resolve_category_seat_types()  → pick OPEN / OBC-NCL / SC / ST / EWS rows
  ├── apply CATEGORY_RANK_RELAXATION if category data absent
  ├── filter by Gender (Gender-Neutral + Female-only if female)
  ├── filter by IIT vs NIT/IIIT (Advanced rank vs Mains rank)
  ├── filter by branch interests (INTEREST_BRANCH_MAP keyword matching)
  └── filter by Quota (HS/OS/AI/GO/JK/LA) per institute and home state
        │
        ▼
For each row: compute_probability(student_rank, opening, closing, institute)
        │
        ▼
Sort by: state_priority → inst_type → safe_zone → preference_score → probability
        │
        ▼
Return grouped results: { IIT: [...], NIT: [...], IIIT: [...], Other: [...] }
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Web Framework | FastAPI 0.111 + Uvicorn | Async HTTP + WebSocket server |
| Templating | Jinja2 | Single-page HTML rendering |
| Data | Pandas + OpenPyXL | JoSAA cutoff Excel parsing |
| AI | Groq (LLaMA-3.3-70B-Versatile) | College counselling, rank analysis, insights |
| AI (fallback) | Google Gemini | Secondary LLM client |
| Database | TiDB Cloud (MySQL-compatible) | All structured data, caching |
| Media Storage | MEGA Cloud | Binary files (images, videos, docs) |
| Live Scraping | Selenium + Perplexity.ai | Real-time college info |
| Validation | Pydantic v2 | Request/response models |
| Connection Pool | mysql-connector-python | 10-connection TiDB pool |

---

## Core Algorithm: College Recommendation Engine

The heart of the system is `filter_options(profile)` in `main.py`. It executes a multi-stage pipeline:

### Stage 1 — Category Resolution

```python
seat_types, is_exact = resolve_category_seat_types(category, AVAILABLE_SEAT_TYPES_IN_DATA)
```

The function maps user-friendly strings (`"obc"`, `"ews"`, `"sc"`) to canonical JoSAA seat type names (`"OBC-NCL"`, `"GEN-EWS"`, `"SC"`). If the loaded dataset only has `OPEN` rows, it falls back to OPEN and sets `is_exact = False`, triggering rank scaling:

```python
if not is_exact and relax != 1.0:
    df["Closing Rank"] = (df["Closing Rank"] * relax).round(0).astype(int)
    df["Opening Rank"] = (df["Opening Rank"] * relax).round(0).astype(int)
```

Relaxation multipliers (applied to OPEN closing rank):

| Category | Multiplier | Rationale |
|----------|-----------|-----------|
| OPEN | 1.00× | Baseline |
| GEN-EWS | 1.15× | ~15% more ranks eligible |
| OBC-NCL | 1.35× | ~35% more ranks eligible |
| SC | 2.80× | Cutoffs typically 2.5–3× OPEN |
| ST | 4.50× | Cutoffs typically 4–5× OPEN |

### Stage 2 — Gender Filter

Male students see only `Gender-Neutral` rows. Female students see both `Gender-Neutral` and `Female-only (including Supernumerary)` rows, giving them access to the extra supernumerary seats mandated by JoSAA.

### Stage 3 — IIT vs NIT/IIIT Split

```python
df = df[df["Institute"].apply(lambda inst: has_adv if is_iit(inst) else has_mains)]
```

IITs use the JEE Advanced rank. NITs, IIITs, and Other CFTIs use the JEE Mains CRL rank. Classification uses regex patterns on institute names:

```python
_IIT_PAT  = re.compile(r'\bindian institute of technology\b|\biit\b', re.I)
_NIT_PAT  = re.compile(r'\bnational institute of technology\b|\bnit\b', re.I)
_IIIT_PAT = re.compile(r'\bindian institute of information technology\b|\biiit\b', re.I)
```

### Stage 4 — Branch / Interest Filter

If the student has declared interests (e.g., `["coding", "electronics"]`), only programs whose name matches the interest category's keywords are retained. The matching uses **word-boundary-aware regex** to avoid false positives (e.g., `"cs"` must not match inside `"mathematics"`):

```python
def _kw_matches(kw: str, text: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])"
    return bool(re.search(pattern, text))
```

### Stage 5 — Quota Filter

For each institute, `get_applicable_quotas()` determines which quota(s) the student qualifies for:

- **IITs** always use `AI` (All India) quota.
- **NITs** use `HS` (Home State) for the NIT located in the student's home state, `OS` (Other State) for all others.
- **Special states** (Goa → `GO`, J&K → `JK`, Ladakh → `LA`) get their respective quotas.

The home-state NIT mapping covers all 31 states/UTs with an NIT.

### Stage 6 — Rank Eligibility Cut

Only rows where `student_rank ≤ closing_rank` are kept. This ensures every returned option is technically reachable for the student.

### Stage 7 — Probability Scoring and Sorting

Each record is assigned a probability and a label, then sorted by a composite key:

```python
records.sort(key=lambda x: (
    x["state_priority"],      # 0=home state, 1=neighbor, 2=other, 3=unknown
    INST_ORDER[x["inst_type"]],  # IIT=0, NIT=1, IIIT=2, Other=3
    0 if x["in_safe_zone"] else 1,
    -x["preference_score"],   # higher interest match = higher priority
    -x["chance_probability"],
    x["closing_rank"],
))
```

---

## Rank Predictor Algorithm

The `/api/rank-predictor` endpoint converts between marks, percentile, and rank using **piecewise linear interpolation** over empirical lookup tables derived from JoSAA official data.

### Interpolation Function

```python
def _interpolate(table, x, x_col=0, y_col=1) -> float:
    # Clamps x to table bounds, then finds the two neighbouring rows
    # and performs linear interpolation between them.
    t = (x - x0) / (x1 - x0)
    return ys[i] + t * (ys[i+1] - ys[i])
```

### Conversion Chains

```
JEE Mains Marks (0–300)
        │  marks_to_percentile_mains()
        ▼
JEE Mains Percentile (0–100)
        │  percentile_to_rank()
        ▼
JEE Mains CRL Rank

JEE Advanced Marks (0–360)
        │  marks_to_rank_advanced()
        ▼
JEE Advanced Rank
```

### Category-Adjusted Effective Rank

```python
equivalent_open_rank = int(round(mains_rank_final / relax))
```

An OBC-NCL student at rank 2,700 effectively competes for seats with an OPEN closing rank of ~2,000 (2700 ÷ 1.35). This is the conceptual "equivalent open rank" used for comparison.

### Branch-Chance Matrix (`predict_branch_chances`)

Scans the entire cutoff dataset, groups programs into interest buckets (coding, electronics, core_engg, etc.), computes admission probability for each program, and aggregates:

```
branch_category → [avg_probability, best_probability, count_of_options, chance_label]
```

---

## Category / Caste Reservation System

All five JoSAA reservation categories are supported:

| Category | Seat % | Closing Rank Factor |
|----------|--------|-------------------|
| OPEN | ~50.5% | 1.0× |
| OBC-NCL | 27% | ~1.35× |
| GEN-EWS | 10% | ~1.15× |
| SC | 15% | ~2.80× |
| ST | 7.5% | ~4.50× |

The `/api/category-info` endpoint returns full eligibility criteria, income limits, certificate requirements, and quota notes for each category.

**Fallback behaviour:** When the dataset contains only `OPEN` rows, the system estimates category cutoffs by scaling OPEN closing ranks with the relaxation factor. A `category_note` warning is included in all API responses to inform the user that values are estimated.

---

## Quota & Home-State Logic

```
get_applicable_quotas(home_state, institute, selected_quota)
   │
   ├── IIT? → return ["AI"]
   │
   ├── Special state quota? (Goa/GO, J&K/JK, Ladakh/LA)
   │       → return [special_quota]
   │
   ├── Is this the home-state NIT?
   │       → return ["HS"]
   │
   ├── Has OS quota?
   │       → return ["OS"]
   │
   └── Fallback → return ["AI"] or all available quotas
```

The `STATE_NIT_MAP` maps every Indian state/UT to its corresponding NIT(s). The `STATE_NEIGHBORS` graph is used for the `state_priority` sort key — colleges in neighbouring states are ranked above far-away colleges.

---

## Probability Scoring Engine

`compute_probability(student_rank, opening_rank, closing_rank, institute)` uses a **sigmoid (logistic) curve** with an institute-volatility buffer:

```python
# If student rank is better than opening rank → near-certain admission
if student_rank <= opening_rank:
    boost = min(margin / opening_rank * 10, 4.0)
    return min(95.0 + boost, 99.0)

# Otherwise: sigmoid decay over the rank window + volatility buffer
rank_range = closing_rank - opening_rank
vol_buffer = closing_rank * volatility(institute)  # IIT=8%, NIT=14%, Other=18%
window     = rank_range + vol_buffer
z          = (student_rank - closing_rank) / window
prob       = 100.0 / (1.0 + exp(3.0 * z))
```

Higher volatility buffers for Other institutes reflect greater year-to-year cutoff variability compared to stable IIT cutoffs.

**Chance labels:**

| Probability | Label |
|-------------|-------|
| ≥ 85% | Safe |
| 60–84% | Good |
| 35–59% | Moderate |
| 15–34% | Reach |
| < 15% | Difficult |

---

## Interest-to-Branch Matching

The `INTEREST_BRANCH_MAP` maps 8 interest categories to comprehensive keyword lists:

| Interest Category | Example Keywords |
|------------------|-----------------|
| `coding` | computer science, cse, AI, data science, cyber security, vlsi |
| `electronics` | electrical engineering, ECE, instrumentation, embedded systems |
| `research` | engineering physics, mathematics & computing, statistics |
| `mba` | industrial engineering, operations research, economics |
| `core_engg` | mechanical, civil, chemical, metallurgical, aerospace, biotech |
| `earth_science` | geology, geophysics, applied geophysics |
| `architecture_design` | architecture, planning, design, engineering design |
| `undecided` | (matches all branches — no filter applied) |

`score_branch_for_interests()` assigns +10 points per keyword match, which feeds into the composite sort key as `preference_score`. This ensures that if two colleges have equal probability, the one offering the student's preferred branch ranks higher.

---

## AI Counselor (Groq LLM)

All AI interactions use **LLaMA-3.3-70B-Versatile** via Groq with low-latency inference.

### Endpoints that call the LLM

| Endpoint | Purpose | Max Tokens | Temp |
|----------|---------|-----------|------|
| `POST /api/chat` | Conversational counsellor | 1000 | 0.5 |
| `POST /api/college-detail` | College profile (7 sections) | 800 | 0.3 |
| `POST /api/college-enquiry` | Q&A with scraper context | 700 | 0.4 |
| `POST /api/college-compare` | Side-by-side comparison table | 1200 | 0.4 |
| `POST /api/rank-predictor` | Rank analysis narrative | 900 | 0.4 |
| Community AI tasks | Daily study plan (5 tasks) | 600 | 0.6 |

### System Prompt Design

The counsellor system prompt (`_build_counselor_system`) injects:
- **Student profile** (rank, gender, state, category, interests)
- **Top 15 shortlisted options** in structured text
- **Category note** (if data was estimated)
- **Perplexity live data** (if `use_perplexity=true`)
- **9 behavioural rules** (accuracy-first, IIT vs NIT distinction, quota logic, etc.)
- **Chain-of-thought instruction** (identify intent → cross-check ranks → structure output)

### College Insights Cache

AI-generated college profiles are cached in TiDB (`college_cache` table) with a **7-day TTL**. The cache is bypassed when live scraper data is attached, so fresh Perplexity content always triggers a new LLM call.

---

## StudyCord Community Platform

A Discord-inspired community embedded directly in the main app (no sub-app mounting).

### Features

| Feature | Description |
|---------|-------------|
| **Feed** | Paginated posts (30/page) with rich media support |
| **Reactions** | Emoji toggle reactions per post per user |
| **Comments** | Threaded comments with media support |
| **Study Groups** | Public/private groups with invite codes, expiry (1–12 months) |
| **Group Chat** | Real-time WebSocket messaging per group |
| **Join Requests** | Private groups require owner approval |
| **AI Study Tasks** | LLM-generated daily study plan per group subject |
| **Media Uploads** | Images, videos, documents up to 50 MB via MEGA |

### WebSocket Architecture

```
Client connects → /ws/group/{group_id}/{username}
        │
        ▼
ConnectionManager.connect()  (asyncio.Lock-protected dict)
        │
On message → broadcast(channel, data) to all sockets in channel
        │
On disconnect → remove from channel set, cleanup dead sockets
```

### Group Lifecycle

Groups have configurable expiry (1–12 months). A background task `_group_cleanup_loop()` runs every hour and calls `db_purge_expired_groups()` to cascade-delete all members, messages, and requests for expired groups.

---

## Perplexity Scraper

`p_scraper.py` provides live college information by automating a headless Chrome browser against Perplexity.ai.

### Flow

```
scrape_college_info(college, branch)
   │
   ├── Check TiDB scrape_cache (24h TTL)
   │       └── Cache hit → return immediately
   │
   └── Cache miss → _do_scrape(query)
           │
           ├── Launch headless Chrome (Selenium)
           ├── Navigate to perplexity.ai
           ├── Dismiss cookie banners
           ├── Find search input (4 selector fallbacks)
           ├── Submit structured query
           ├── Wait for answer element (3 selector fallbacks)
           ├── Wait for answer to reach >50 chars
           └── Return raw text
           │
           ▼
       _parse_sections(text)  →  split into 9 named sections
           │
           ▼
       Store in TiDB scrape_cache
```

The structured query requests: admission process, placements, branches, campus, research, student life, alumni, fees, and pros/cons — giving the LLM rich, current context.

---

## Database Schema (TiDB)

All tables are created automatically at startup via `init_db()`.

```
posts            — community feed (id, author, content, media_url, media_type, created_at)
post_reactions   — emoji reactions (post_id, emoji, username) [composite PK]
comments         — post comments  (id, post_id, author, content, media_url)
groups           — study groups   (id, name, code, group_type, expiry_date, created_by)
group_members    — membership     (group_id, username) [composite PK]
group_messages   — chat messages  (id, group_id, author, content, media_url)
group_requests   — join requests  (group_id, username, status)
chat_history     — AI counsellor history per user (username → JSON blob)
college_cache    — AI insights    (college_name → text, cached_at) [7-day TTL]
scrape_cache     — Perplexity results (query_key MD5 → JSON, scraped_at) [24h TTL]
```

**Connection pool:** 10 connections, TLS enforced, `autocommit=True`. All DB operations are synchronous and run in a `ThreadPoolExecutor` (8 workers) so the asyncio event loop is never blocked.

---

## MEGA Media Store

`MegaMediaStore` handles binary file uploads for community posts and messages.

### Upload Flow

```
POST /api/upload-media
   │
   ├── Validate content-type (image/video/document)
   ├── Check file size ≤ 50 MB
   ├── Write to local temp staging dir
   │
   └── _executor.submit(mega.upload_get_link)
           │
           ├── Connect to MEGA (lazy, thread-safe)
           ├── Ensure sc_media/ folder exists
           ├── Delete any existing file with same name
           ├── Upload file
           ├── Get public share link
           └── Cleanup staging file
```

Returns a public MEGA link stored in the `media_url` column of posts/comments/messages.

---

## API Reference

### Advisor Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve index.html |
| GET | `/api/filters` | States, seat types, genders, quotas |
| GET | `/api/category-info?category=OBC-NCL` | Category eligibility and quota details |
| POST | `/api/get-options` | Core recommendation engine |
| POST | `/api/rank-predictor` | Marks/percentile/rank conversion + branch matrix |
| POST | `/api/college-detail` | AI college profile (7 sections) |
| POST | `/api/college-compare` | Side-by-side comparison of 2–4 colleges |
| POST | `/api/college-enquiry` | Q&A with live Perplexity data |
| POST | `/api/chat` | AI counsellor conversation |

### Community Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/posts` | Paginated feed |
| POST | `/api/posts` | Create post |
| DELETE | `/api/posts/{id}` | Delete own post |
| POST | `/api/posts/{id}/react` | Toggle emoji reaction |
| GET | `/api/posts/{id}/comments` | Get comments |
| POST | `/api/posts/{id}/comments` | Add comment |
| GET/POST | `/api/groups` | List / create study groups |
| POST | `/api/groups/join` | Join by invite code |
| POST | `/api/groups/{id}/leave` | Leave group |
| DELETE | `/api/groups/{id}` | Delete group (owner only) |
| GET | `/api/groups/{id}/messages` | Message history |
| POST | `/api/groups/{id}/messages` | Send message |
| POST | `/api/groups/{id}/ai-tasks` | Generate AI study plan |
| POST | `/api/upload-media` | Upload binary file to MEGA |
| WS | `/ws/group/{group_id}/{username}` | Real-time group chat |

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Groq AI
GROQ_API_KEY=gsk_...

# Google Gemini (optional fallback)
GOOGLE_GEMINI_API_KEY=AIza...
GOOGLE_GEMINI_MODEL=gemini-2.0-flash

# TiDB Cloud
TIDB_HOST=gateway01.ap-southeast-1.prod.aws.tidbcloud.com
TIDB_PORT=4000
TIDB_USER=your_user
TIDB_PASSWORD=your_password
TIDB_DB=jee_advisor
TIDB_SSL_CA=                    # Optional: path to CA cert

# MEGA Cloud Storage
MEGA_EMAIL=your@email.com
MEGA_PASSWORD=your_mega_password
```

---

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env file (see above)

# 3. Start the server
uvicorn main:app --reload --port 8000

# 4. Open browser
# http://localhost:8000
```

The app initialises the TiDB schema, purges expired groups, and connects to MEGA automatically on startup via the FastAPI `lifespan` handler.

---

## Data Files

| File | Description |
|------|-------------|
| `data/JEE_2025_Cutoffs.xlsx` | JoSAA 2025 opening/closing ranks for all programs. Required columns: Institute, Academic Program Name, Quota, Seat Type, Gender, Opening Rank, Closing Rank |
| `data/JEE_2025_ALL_128_COLLEGES.xlsx` | College metadata: College Name, Official Website, Location, Category |

The cutoff file is loaded at startup into a Pandas DataFrame. Numeric columns are coerced and NaN rows are dropped. The app will raise a `RuntimeError` at boot if any required column is missing.

---

## Key Design Decisions

1. **Flat router architecture** — all routes are registered directly on the `FastAPI` app (no sub-app mounting) to avoid path-stripping issues with proxies and the Starlette router.

2. **Thread pool for blocking I/O** — TiDB queries, MEGA uploads, and Selenium scraping all run in a shared `ThreadPoolExecutor(max_workers=8)` via `loop.run_in_executor()` to keep the asyncio event loop non-blocking.

3. **Sigmoid probability vs hard cutoff** — instead of a binary "eligible/not eligible" result, the sigmoid model gives students a nuanced view of how close they are to the boundary, with volatility buffers reflecting historical cutoff variation per institute type.

4. **Estimate-and-warn for missing category data** — rather than silently dropping students whose category has no data, the system estimates cutoffs with documented multipliers and always includes a `category_note` in responses.

5. **7-day college insights cache** — LLM calls are expensive; caching college profiles in TiDB avoids redundant API calls while keeping data reasonably fresh.
