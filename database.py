"""
database.py — TiDB Cloud (MySQL-compatible) persistence layer
=============================================================
All application state is stored in TiDB. MEGA is used ONLY for binary
media files (images, videos, documents). JSON blobs (posts, groups,
messages, chat history) live in TiDB tables.

Connection is established once at startup via a connection pool.
All operations are synchronous (run in a thread pool by FastAPI).

Tables created automatically on first run:
  • posts          — community feed posts
  • post_reactions — emoji reactions (many-to-many)
  • comments       — post comments
  • groups         — study groups
  • group_members  — group membership
  • group_messages — group chat messages
  • group_requests — pending join requests (private groups)
  • chat_history   — AI counselor conversation history per user
  • college_cache  — AI-generated college insights cache (TTL 7 days)
  • scrape_cache   — Selenium-scraped college data cache
"""

import os
import json
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

import mysql.connector
from mysql.connector import pooling
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")  # Load environment variables from .env file
# ── Credentials (set in environment) ─────────────────────────────────────────
TIDB_HOST     = os.getenv("TIDB_HOST")
TIDB_PORT     = int(os.getenv("TIDB_PORT", "4000"))
TIDB_USER     = os.getenv("TIDB_USER")
TIDB_PASSWORD = os.getenv("TIDB_PASSWORD")
TIDB_DB       = os.getenv("TIDB_DB")
TIDB_SSL_CA   = os.getenv("TIDB_SSL_CA") # path to CA cert, if required

_pool: Optional[pooling.MySQLConnectionPool] = None
_pool_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
# CONNECTION POOL
# ══════════════════════════════════════════════════════════════════════════════

def _build_pool() -> pooling.MySQLConnectionPool:
    kwargs: Dict[str, Any] = dict(
        pool_name="tidb_pool",
        pool_size=10,
        host=TIDB_HOST,
        port=TIDB_PORT,
        user=TIDB_USER,
        password=TIDB_PASSWORD,
        database=TIDB_DB,
        autocommit=True,
        connection_timeout=15,
        charset="utf8mb4",
        collation="utf8mb4_unicode_ci",
    )
    if TIDB_SSL_CA:
        kwargs["ssl_ca"]     = TIDB_SSL_CA
        kwargs["ssl_verify_cert"] = True
    else:
        # TiDB Cloud requires TLS; set ssl_disabled=False without a CA cert
        # and the driver will use the system trust store.
        kwargs["ssl_disabled"] = False
    return pooling.MySQLConnectionPool(**kwargs)


def get_pool() -> pooling.MySQLConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = _build_pool()
    return _pool


@contextmanager
def get_conn():
    pool = get_pool()
    conn = pool.get_connection()
    try:
        yield conn
    finally:
        conn.close()


def execute(sql: str, params: tuple = (), fetch: str = "none") -> Any:
    """
    Run a single SQL statement.
    fetch = "one" | "all" | "none"
    Returns lastrowid for INSERT/UPDATE when fetch="none".
    """
    with get_conn() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        if fetch == "one":
            return cur.fetchone()
        if fetch == "all":
            return cur.fetchall()
        return cur.lastrowid


def executemany(sql: str, params_list: List[tuple]) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.executemany(sql, params_list)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════

DDL = [
    # ── posts ─────────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS posts (
        id           VARCHAR(36)  NOT NULL PRIMARY KEY,
        author       VARCHAR(100) NOT NULL,
        avatar_color VARCHAR(20)  DEFAULT '#5865F2',
        content      TEXT,
        media_url    TEXT,
        media_type   VARCHAR(20)  DEFAULT '',
        created_at   DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
        comment_count INT         DEFAULT 0,
        INDEX idx_posts_created (created_at DESC)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,

    # ── post_reactions ────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS post_reactions (
        post_id  VARCHAR(36)  NOT NULL,
        emoji    VARCHAR(10)  NOT NULL,
        username VARCHAR(100) NOT NULL,
        PRIMARY KEY (post_id, emoji, username),
        INDEX idx_rxn_post (post_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,

    # ── comments ──────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS comments (
        id           VARCHAR(36)  NOT NULL PRIMARY KEY,
        post_id      VARCHAR(36)  NOT NULL,
        author       VARCHAR(100) NOT NULL,
        avatar_color VARCHAR(20)  DEFAULT '#5865F2',
        content      TEXT,
        media_url    TEXT,
        media_type   VARCHAR(20)  DEFAULT '',
        created_at   DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
        INDEX idx_cmt_post (post_id),
        INDEX idx_cmt_created (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,

    # ── groups ────────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS `groups` (
        id             VARCHAR(36)  NOT NULL PRIMARY KEY,
        name           VARCHAR(200) NOT NULL,
        subject        VARCHAR(200) DEFAULT '',
        code           VARCHAR(10)  NOT NULL UNIQUE,
        group_type     VARCHAR(20)  DEFAULT 'public',
        expiry_months  INT          DEFAULT 3,
        expiry_date    DATETIME     NOT NULL,
        created_at     DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
        created_by     VARCHAR(100) NOT NULL,
        INDEX idx_grp_code (code),
        INDEX idx_grp_expiry (expiry_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,

    # ── group_members ─────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS group_members (
        group_id     VARCHAR(36)  NOT NULL,
        username     VARCHAR(100) NOT NULL,
        avatar_color VARCHAR(20)  DEFAULT '#5865F2',
        joined_at    DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
        PRIMARY KEY (group_id, username),
        INDEX idx_gmbr_user (username)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,

    # ── group_messages ────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS group_messages (
        id           VARCHAR(36)  NOT NULL PRIMARY KEY,
        group_id     VARCHAR(36)  NOT NULL,
        author       VARCHAR(100) NOT NULL,
        avatar_color VARCHAR(20)  DEFAULT '#5865F2',
        content      TEXT,
        media_url    TEXT,
        media_type   VARCHAR(20)  DEFAULT '',
        created_at   DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
        INDEX idx_gmsg_grp (group_id, created_at DESC)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,

    # ── group_requests ────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS group_requests (
        group_id     VARCHAR(36)  NOT NULL,
        username     VARCHAR(100) NOT NULL,
        avatar_color VARCHAR(20)  DEFAULT '#5865F2',
        requested_at DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
        status       VARCHAR(20)  DEFAULT 'pending',
        PRIMARY KEY (group_id, username)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,

    # ── chat_history ──────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS chat_history (
        username    VARCHAR(100) NOT NULL PRIMARY KEY,
        history_json LONGTEXT    NOT NULL,
        updated_at  DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                    ON UPDATE CURRENT_TIMESTAMP(3)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,

    # ── college_cache ─────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS college_cache (
        college_name VARCHAR(300) NOT NULL PRIMARY KEY,
        insights     LONGTEXT     NOT NULL,
        cached_at    DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,

    # ── scrape_cache ──────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS scrape_cache (
        query_key    VARCHAR(300) NOT NULL PRIMARY KEY,
        result_json  LONGTEXT     NOT NULL,
        scraped_at   DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
]


def init_db() -> None:
    """Create all tables if they don't exist. Called once at startup."""
    print(f"[DB] Initialising TiDB schema on database '{TIDB_DB}' …")
    try:
        for idx, ddl in enumerate(DDL, start=1):
            first_line = ddl.strip().splitlines()[0] if ddl.strip() else "<empty>"
            print(f"[DB] Executing DDL {idx}/{len(DDL)}: {first_line}")
            execute(ddl.strip())
            print(f"[DB] Completed DDL {idx}/{len(DDL)}")
    except mysql.connector.Error as exc:
        raise RuntimeError(
            "TiDB schema initialization failed. "
            f"Please verify TIDB_DB is set to a writable application database, "
            "and that the configured user has CREATE TABLE privileges. "
            f"Failed on DDL {idx}/{len(DDL)}: {first_line}. Original error: {exc}") from exc
    print("[DB] Schema ready.")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — timestamp
# ══════════════════════════════════════════════════════════════════════════════

def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _dt_to_str(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# ══════════════════════════════════════════════════════════════════════════════
# POSTS
# ══════════════════════════════════════════════════════════════════════════════

def db_create_post(post: dict) -> dict:
    execute(
        """INSERT INTO posts (id, author, avatar_color, content, media_url, media_type, created_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (
            post["id"], post["author"], post["avatar_color"],
            post["content"], post["media_url"], post["media_type"],
            post["timestamp"],
        ),
    )
    return post


def db_get_posts(page: int = 0, per_page: int = 30) -> List[dict]:
    rows = execute(
        "SELECT * FROM posts ORDER BY created_at DESC LIMIT %s OFFSET %s",
        (per_page, page * per_page),
        fetch="all",
    ) or []
    result = []
    for row in rows:
        row["timestamp"] = _dt_to_str(row["created_at"]) if isinstance(row["created_at"], datetime) else str(row["created_at"])
        row["reactions"]     = _load_reactions(row["id"])
        row["comment_count"] = row.get("comment_count", 0)
        result.append(row)
    return result


def db_count_posts() -> int:
    row = execute("SELECT COUNT(*) AS cnt FROM posts", fetch="one")
    return (row or {}).get("cnt", 0)


def db_delete_post(post_id: str, author: str) -> Optional[dict]:
    row = execute("SELECT * FROM posts WHERE id=%s", (post_id,), fetch="one")
    if not row:
        return None
    if row["author"] != author:
        raise PermissionError("not your post")
    execute("DELETE FROM post_reactions WHERE post_id=%s", (post_id,))
    execute("DELETE FROM comments WHERE post_id=%s", (post_id,))
    execute("DELETE FROM posts WHERE id=%s", (post_id,))
    return row


def db_get_post(post_id: str) -> Optional[dict]:
    return execute("SELECT * FROM posts WHERE id=%s", (post_id,), fetch="one")


# ── Reactions ─────────────────────────────────────────────────────────────────

def _load_reactions(post_id: str) -> dict:
    rows = execute(
        "SELECT emoji, username FROM post_reactions WHERE post_id=%s",
        (post_id,), fetch="all",
    ) or []
    out: Dict[str, List[str]] = {}
    for r in rows:
        out.setdefault(r["emoji"], []).append(r["username"])
    return out


def db_toggle_reaction(post_id: str, emoji: str, username: str) -> dict:
    existing = execute(
        "SELECT 1 FROM post_reactions WHERE post_id=%s AND emoji=%s AND username=%s",
        (post_id, emoji, username), fetch="one",
    )
    if existing:
        execute(
            "DELETE FROM post_reactions WHERE post_id=%s AND emoji=%s AND username=%s",
            (post_id, emoji, username),
        )
    else:
        execute(
            "INSERT IGNORE INTO post_reactions (post_id, emoji, username) VALUES (%s,%s,%s)",
            (post_id, emoji, username),
        )
    return _load_reactions(post_id)


# ── Comments ──────────────────────────────────────────────────────────────────

def db_add_comment(comment: dict) -> dict:
    execute(
        """INSERT INTO comments (id, post_id, author, avatar_color, content, media_url, media_type, created_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        (
            comment["id"], comment["post_id"], comment["author"],
            comment["avatar_color"], comment["content"],
            comment["media_url"], comment["media_type"], comment["timestamp"],
        ),
    )
    cnt = execute(
        "SELECT COUNT(*) AS cnt FROM comments WHERE post_id=%s",
        (comment["post_id"],), fetch="one",
    )["cnt"]
    execute("UPDATE posts SET comment_count=%s WHERE id=%s", (cnt, comment["post_id"]))
    comment["comment_count"] = cnt
    return comment


def db_get_comments(post_id: str) -> List[dict]:
    rows = execute(
        "SELECT * FROM comments WHERE post_id=%s ORDER BY created_at",
        (post_id,), fetch="all",
    ) or []
    for r in rows:
        r["timestamp"] = _dt_to_str(r["created_at"]) if isinstance(r["created_at"], datetime) else str(r["created_at"])
    return rows


def db_delete_comment(post_id: str, comment_id: str, author: str) -> Optional[dict]:
    row = execute("SELECT * FROM comments WHERE id=%s AND post_id=%s", (comment_id, post_id), fetch="one")
    if not row:
        return None
    if row["author"] != author:
        raise PermissionError("not your comment")
    execute("DELETE FROM comments WHERE id=%s", (comment_id,))
    cnt = execute("SELECT COUNT(*) AS cnt FROM comments WHERE post_id=%s", (post_id,), fetch="one")["cnt"]
    execute("UPDATE posts SET comment_count=%s WHERE id=%s", (cnt, post_id))
    return row


# ══════════════════════════════════════════════════════════════════════════════
# GROUPS
# ══════════════════════════════════════════════════════════════════════════════

def db_create_group(group: dict, creator_username: str, creator_color: str) -> dict:
    execute(
        """INSERT INTO `groups` (id, name, subject, code, group_type, expiry_months, expiry_date, created_at, created_by)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (
            group["id"], group["name"], group["subject"], group["code"],
            group["group_type"], group["expiry_months"], group["expiry_date"],
            group["created_at"], group["created_by"],
        ),
    )
    execute(
        "INSERT INTO group_members (group_id, username, avatar_color, joined_at) VALUES (%s,%s,%s,%s)",
        (group["id"], creator_username, creator_color, group["created_at"]),
    )
    return group


def db_get_groups_public(q: str = "") -> List[dict]:
    if q:
        rows = execute(
            """SELECT g.*, COUNT(gm.username) AS member_count
               FROM `groups` g
               LEFT JOIN group_members gm ON gm.group_id=g.id
               WHERE g.group_type='public' AND (g.name LIKE %s OR g.subject LIKE %s)
               GROUP BY g.id ORDER BY g.created_at DESC LIMIT 20""",
            (f"%{q}%", f"%{q}%"), fetch="all",
        ) or []
    else:
        rows = execute(
            """SELECT g.*, COUNT(gm.username) AS member_count
               FROM `groups` g
               LEFT JOIN group_members gm ON gm.group_id=g.id
               WHERE g.group_type='public'
               GROUP BY g.id ORDER BY g.created_at DESC LIMIT 20""",
            fetch="all",
        ) or []
    out = []
    for r in rows:
        r.pop("code", None)  # never expose code in public listing
        r["expiry_date"] = _dt_to_str(r["expiry_date"]) if isinstance(r["expiry_date"], datetime) else str(r["expiry_date"])
        r["created_at"]  = _dt_to_str(r["created_at"])  if isinstance(r["created_at"],  datetime) else str(r["created_at"])
        out.append(r)
    return out
def db_get_my_groups(username: str) -> List[dict]:
    rows = execute(
        """SELECT g.*, COUNT(gm2.username) AS member_count,
                  COALESCE(MAX(reqs.pending_requests), 0) AS pending_requests
           FROM `groups` g
           JOIN group_members gm ON gm.group_id=g.id AND gm.username=%s
           LEFT JOIN group_members gm2 ON gm2.group_id=g.id
           LEFT JOIN (
             SELECT group_id, COUNT(*) AS pending_requests
             FROM group_requests
             GROUP BY group_id
           ) reqs ON reqs.group_id=g.id
           GROUP BY g.id ORDER BY g.created_at DESC""",
        (username,), fetch="all",
    ) or []
    out = []
    for r in rows:
        is_creator = r.get("created_by") == username
        if not is_creator:
            r.pop("code", None)
        r["expiry_date"] = _dt_to_str(r["expiry_date"]) if isinstance(r["expiry_date"], datetime) else str(r["expiry_date"])
        r["created_at"]  = _dt_to_str(r["created_at"])  if isinstance(r["created_at"],  datetime) else str(r["created_at"])
        out.append(r)
    return out


def db_get_group(group_id: str) -> Optional[dict]:
    return execute("SELECT * FROM `groups` WHERE id=%s", (group_id,), fetch="one")


def db_get_group_by_code(code: str) -> Optional[dict]:
    return execute("SELECT * FROM `groups` WHERE code=%s", (code,), fetch="one")


def db_delete_group(group_id: str, owner: str) -> bool:
    g = db_get_group(group_id)
    if not g or g["created_by"] != owner:
        return False
    execute("DELETE FROM group_members  WHERE group_id=%s", (group_id,))
    execute("DELETE FROM group_messages WHERE group_id=%s", (group_id,))
    execute("DELETE FROM group_requests WHERE group_id=%s", (group_id,))
    execute("DELETE FROM `groups` WHERE id=%s", (group_id,))
    return True


def db_purge_expired_groups() -> int:
    rows = execute(
        "SELECT id FROM `groups` WHERE expiry_date < NOW()",
        fetch="all",
    ) or []
    for r in rows:
        gid = r["id"]
        execute("DELETE FROM group_members  WHERE group_id=%s", (gid,))
        execute("DELETE FROM group_messages WHERE group_id=%s", (gid,))
        execute("DELETE FROM group_requests WHERE group_id=%s", (gid,))
    if rows:
        execute("DELETE FROM `groups` WHERE expiry_date < NOW()")
    return len(rows)


# ── Members ───────────────────────────────────────────────────────────────────

def db_is_member(group_id: str, username: str) -> bool:
    return bool(execute(
        "SELECT 1 FROM group_members WHERE group_id=%s AND username=%s",
        (group_id, username), fetch="one",
    ))


def db_add_member(group_id: str, username: str, avatar_color: str) -> None:
    execute(
        "INSERT IGNORE INTO group_members (group_id, username, avatar_color) VALUES (%s,%s,%s)",
        (group_id, username, avatar_color),
    )


def db_remove_member(group_id: str, username: str) -> None:
    execute("DELETE FROM group_members WHERE group_id=%s AND username=%s", (group_id, username))


def db_get_members(group_id: str) -> List[dict]:
    rows = execute(
        "SELECT * FROM group_members WHERE group_id=%s ORDER BY joined_at",
        (group_id,), fetch="all",
    ) or []
    for r in rows:
        r["joined_at"] = _dt_to_str(r["joined_at"]) if isinstance(r["joined_at"], datetime) else str(r["joined_at"])
    return rows


# ── Join Requests ─────────────────────────────────────────────────────────────

def db_has_request(group_id: str, username: str) -> bool:
    return bool(execute(
        "SELECT 1 FROM group_requests WHERE group_id=%s AND username=%s",
        (group_id, username), fetch="one",
    ))


def db_add_request(group_id: str, username: str, avatar_color: str) -> None:
    execute(
        "INSERT IGNORE INTO group_requests (group_id, username, avatar_color) VALUES (%s,%s,%s)",
        (group_id, username, avatar_color),
    )


def db_get_requests(group_id: str) -> List[dict]:
    rows = execute(
        "SELECT * FROM group_requests WHERE group_id=%s ORDER BY requested_at",
        (group_id,), fetch="all",
    ) or []
    for r in rows:
        r["requested_at"] = _dt_to_str(r["requested_at"]) if isinstance(r["requested_at"], datetime) else str(r["requested_at"])
    return rows


def db_remove_request(group_id: str, username: str) -> None:
    execute("DELETE FROM group_requests WHERE group_id=%s AND username=%s", (group_id, username))


# ── Group Messages ────────────────────────────────────────────────────────────

def db_send_message(msg: dict) -> dict:
    execute(
        """INSERT INTO group_messages (id, group_id, author, avatar_color, content, media_url, media_type, created_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        (
            msg["id"], msg["group_id"], msg["author"], msg["avatar_color"],
            msg["content"], msg["media_url"], msg["media_type"], msg["timestamp"],
        ),
    )
    return msg


def db_get_messages(group_id: str, after: str = "") -> List[dict]:
    if after:
        rows = execute(
            "SELECT * FROM group_messages WHERE group_id=%s AND created_at > %s ORDER BY created_at",
            (group_id, after), fetch="all",
        ) or []
    else:
        rows = execute(
            "SELECT * FROM group_messages WHERE group_id=%s ORDER BY created_at",
            (group_id,), fetch="all",
        ) or []
    for r in rows:
        r["timestamp"] = _dt_to_str(r["created_at"]) if isinstance(r["created_at"], datetime) else str(r["created_at"])
    return rows


def db_delete_message(group_id: str, msg_id: str, author: str) -> Optional[dict]:
    row = execute(
        "SELECT * FROM group_messages WHERE id=%s AND group_id=%s",
        (msg_id, group_id), fetch="one",
    )
    if not row:
        return None
    if row["author"] != author:
        raise PermissionError("not your message")
    execute("DELETE FROM group_messages WHERE id=%s", (msg_id,))
    return row


# ══════════════════════════════════════════════════════════════════════════════
# CHAT HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def db_get_chat_history(username: str) -> List[dict]:
    row = execute(
        "SELECT history_json FROM chat_history WHERE username=%s",
        (username,), fetch="one",
    )
    if not row:
        return []
    try:
        return json.loads(row["history_json"]) or []
    except Exception:
        return []


def db_set_chat_history(username: str, history: List[dict]) -> None:
    blob = json.dumps(history, ensure_ascii=False)
    execute(
        """INSERT INTO chat_history (username, history_json) VALUES (%s,%s)
           ON DUPLICATE KEY UPDATE history_json=%s, updated_at=CURRENT_TIMESTAMP(3)""",
        (username, blob, blob),
    )


def db_clear_chat_history(username: str) -> None:
    execute(
        "INSERT INTO chat_history (username, history_json) VALUES (%s,'[]') ON DUPLICATE KEY UPDATE history_json='[]'",
        (username,),
    )


# ══════════════════════════════════════════════════════════════════════════════
# COLLEGE CACHE
# ══════════════════════════════════════════════════════════════════════════════
CACHE_TTL_DAYS = 7


def db_get_college_insights(college_name: str) -> Optional[str]:
    row = execute(
        "SELECT insights, cached_at FROM college_cache WHERE college_name=%s",
        (college_name,), fetch="one",
    )
    if not row:
        return None
    cached_at = row["cached_at"]
    if isinstance(cached_at, datetime):
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - cached_at > timedelta(days=CACHE_TTL_DAYS):
            return None
    return row["insights"]


def db_set_college_insights(college_name: str, insights: str) -> None:
    execute(
        """INSERT INTO college_cache (college_name, insights) VALUES (%s,%s)
           ON DUPLICATE KEY UPDATE insights=%s, cached_at=CURRENT_TIMESTAMP(3)""",
        (college_name, insights, insights),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SCRAPE CACHE
# ══════════════════════════════════════════════════════════════════════════════
SCRAPE_TTL_DAYS = 1


def db_get_scrape(query_key: str) -> Optional[dict]:
    row = execute(
        "SELECT result_json, scraped_at FROM scrape_cache WHERE query_key=%s",
        (query_key,), fetch="one",
    )
    if not row:
        return None
    scraped_at = row["scraped_at"]
    if isinstance(scraped_at, datetime):
        if scraped_at.tzinfo is None:
            scraped_at = scraped_at.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - scraped_at > timedelta(days=SCRAPE_TTL_DAYS):
            return None
    try:
        return json.loads(row["result_json"])
    except Exception:
        return None


def db_set_scrape(query_key: str, result: dict) -> None:
    blob = json.dumps(result, ensure_ascii=False)
    execute(
        """INSERT INTO scrape_cache (query_key, result_json) VALUES (%s,%s)
           ON DUPLICATE KEY UPDATE result_json=%s, scraped_at=CURRENT_TIMESTAMP(3)""",
        (query_key, blob, blob),
    )
