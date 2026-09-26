"""Data access for the platform: users, orgs, projects, API keys, credits and usage.

Plain sqlite3, one short-lived connection per call. Every function returns plain
dicts so callers (API gateway, dashboard) never touch SQL.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from minilab.settings import get_settings

MICROS_PER_USD = 1_000_000
SESSION_TTL_S = 30 * 24 * 3600
API_KEY_PREFIX = "sk-mini-"
BODY_LOG_LIMIT = 8_000  # chars of request/response JSON kept in the logs

_db_path: str | None = None


def configure(path: str | Path | None) -> None:
    """Point the store at a database file (None: back to settings.db_path)."""
    global _db_path
    _db_path = str(path) if path is not None else None


def _path() -> str:
    return _db_path or get_settings().db_path


@contextmanager
def connect():
    conn = sqlite3.connect(_path(), timeout=30, isolation_level=None)  # autocommit; explicit BEGIN for txns
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction():
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise


def init_db() -> None:
    with connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript((Path(__file__).parent / "schema.sql").read_text())


def _now() -> int:
    return int(time.time())


def _id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def _row(r: sqlite3.Row | None) -> dict | None:
    return dict(r) if r is not None else None


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def usd_to_micros(usd: float) -> int:
    return round(usd * MICROS_PER_USD)


def format_usd(micros: int, digits: int = 2) -> str:
    sign = "-" if micros < 0 else ""
    return f"{sign}${abs(micros) / MICROS_PER_USD:,.{digits}f}"


# ---- users & sessions -------------------------------------------------------

def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def _check_password(password: str, stored: str) -> bool:
    _, salt_hex, _ = stored.split("$")
    return hmac.compare_digest(_hash_password(password, bytes.fromhex(salt_hex)), stored)


def create_user(email: str, password: str, name: str = "") -> dict:
    """Create a user. Raises ValueError if the email is taken."""
    email = email.strip().lower()
    user = {"id": _id("user"), "email": email, "name": name.strip(), "created_at": _now()}
    try:
        with connect() as conn:
            conn.execute(
                "INSERT INTO users (id, email, name, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (user["id"], email, user["name"], _hash_password(password), user["created_at"]),
            )
    except sqlite3.IntegrityError:
        raise ValueError("An account with this email already exists.") from None
    return user


def authenticate(email: str, password: str) -> dict | None:
    with connect() as conn:
        r = conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    if r is None:
        _hash_password(password)  # same cost as a real check: timing doesn't reveal unknown emails
        return None
    if not _check_password(password, r["password_hash"]):
        return None
    user = dict(r)
    user.pop("password_hash")
    return user


def get_user(user_id: str) -> dict | None:
    with connect() as conn:
        return _row(conn.execute("SELECT id, email, name, created_at FROM users WHERE id = ?", (user_id,)).fetchone())


def create_session(user_id: str) -> str:
    """Returns the session token to put in a cookie (only its hash is stored)."""
    token = secrets.token_urlsafe(32)
    now = _now()
    with connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (_sha256(token), user_id, now, now + SESSION_TTL_S),
        )
    return token


def get_user_by_session(token: str | None) -> dict | None:
    if not token:
        return None
    with connect() as conn:
        return _row(conn.execute(
            "SELECT u.id, u.email, u.name, u.created_at FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_at > ?",
            (_sha256(token), _now()),
        ).fetchone())


def delete_session(token: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_sha256(token),))


# ---- orgs & projects --------------------------------------------------------

def create_org(name: str, owner_id: str, signup_credit_usd: float | None = None) -> dict:
    """Create an org with a 'Default project' and the signup credit grant."""
    if signup_credit_usd is None:
        signup_credit_usd = get_settings().signup_credit_usd
    org = {"id": _id("org"), "name": name.strip() or "Personal", "created_at": _now()}
    with transaction() as conn:
        conn.execute("INSERT INTO orgs (id, name, created_at) VALUES (?, ?, ?)", (org["id"], org["name"], org["created_at"]))
        conn.execute("INSERT INTO org_members (org_id, user_id, role) VALUES (?, ?, 'owner')", (org["id"], owner_id))
        conn.execute(
            "INSERT INTO projects (id, org_id, name, created_at) VALUES (?, ?, 'Default project', ?)",
            (_id("proj"), org["id"], org["created_at"]),
        )
    if signup_credit_usd > 0:
        add_credits(org["id"], usd_to_micros(signup_credit_usd), "grant", f"signup:{org['id']}", "Free signup credits")
    return get_org(org["id"])


def get_org(org_id: str) -> dict | None:
    with connect() as conn:
        return _row(conn.execute("SELECT * FROM orgs WHERE id = ?", (org_id,)).fetchone())


def list_user_orgs(user_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT o.*, m.role FROM orgs o JOIN org_members m ON m.org_id = o.id WHERE m.user_id = ? ORDER BY o.created_at",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def is_member(org_id: str, user_id: str) -> bool:
    with connect() as conn:
        return conn.execute("SELECT 1 FROM org_members WHERE org_id = ? AND user_id = ?", (org_id, user_id)).fetchone() is not None


def create_project(org_id: str, name: str) -> dict:
    project = {"id": _id("proj"), "org_id": org_id, "name": name.strip() or "Untitled project", "created_at": _now()}
    with connect() as conn:
        conn.execute("INSERT INTO projects (id, org_id, name, created_at) VALUES (:id, :org_id, :name, :created_at)", project)
    return project


def list_projects(org_id: str) -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM projects WHERE org_id = ? ORDER BY created_at", (org_id,))]


def get_project(project_id: str) -> dict | None:
    with connect() as conn:
        return _row(conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone())


# ---- API keys ---------------------------------------------------------------

def create_api_key(org_id: str, project_id: str, name: str, created_by: str | None = None,
                   spend_limit_usd: float | None = None, rpm_limit: int | None = None,
                   tpm_limit: int | None = None) -> tuple[dict, str]:
    """Create a key. Returns (key row, plaintext secret). The secret is shown to the user once."""
    secret = API_KEY_PREFIX + secrets.token_urlsafe(24)
    row = {
        "id": _id("key"), "org_id": org_id, "project_id": project_id, "name": name.strip() or "Secret key",
        "key_hash": _sha256(secret), "key_hint": f"{secret[:12]}...{secret[-4:]}", "created_by": created_by,
        "created_at": _now(),
        "spend_limit_micros": usd_to_micros(spend_limit_usd) if spend_limit_usd is not None else None,
        "rpm_limit": rpm_limit, "tpm_limit": tpm_limit,
    }
    with connect() as conn:
        conn.execute(
            "INSERT INTO api_keys (id, org_id, project_id, name, key_hash, key_hint, created_by, created_at, "
            "spend_limit_micros, rpm_limit, tpm_limit) VALUES (:id, :org_id, :project_id, :name, :key_hash, "
            ":key_hint, :created_by, :created_at, :spend_limit_micros, :rpm_limit, :tpm_limit)",
            row,
        )
    row.pop("key_hash")
    return row, secret


def list_api_keys(org_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT k.id, k.org_id, k.project_id, p.name AS project_name, k.name, k.key_hint, k.created_by, "
            "k.created_at, k.last_used_at, k.revoked_at, k.spend_micros, k.spend_limit_micros, k.rpm_limit, "
            "k.tpm_limit FROM api_keys k JOIN projects p ON p.id = k.project_id WHERE k.org_id = ? "
            "ORDER BY k.created_at DESC",
            (org_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def revoke_api_key(org_id: str, key_id: str) -> bool:
    with connect() as conn:
        cur = conn.execute(
            "UPDATE api_keys SET revoked_at = ? WHERE id = ? AND org_id = ? AND revoked_at IS NULL",
            (_now(), key_id, org_id),
        )
    return cur.rowcount == 1


def lookup_api_key(secret: str) -> dict | None:
    """Resolve an active key from its plaintext secret. Includes the org balance."""
    if not secret or not secret.startswith(API_KEY_PREFIX):
        return None
    with connect() as conn:
        return _row(conn.execute(
            "SELECT k.id, k.org_id, k.project_id, k.name, k.key_hint, k.spend_micros, k.spend_limit_micros, "
            "k.rpm_limit, k.tpm_limit, o.balance_micros FROM api_keys k JOIN orgs o ON o.id = k.org_id "
            "WHERE k.key_hash = ? AND k.revoked_at IS NULL",
            (_sha256(secret),),
        ).fetchone())


# ---- credits ----------------------------------------------------------------

def get_balance_micros(org_id: str) -> int:
    with connect() as conn:
        r = conn.execute("SELECT balance_micros FROM orgs WHERE id = ?", (org_id,)).fetchone()
    return r["balance_micros"] if r else 0


def _ledger_insert(conn, org_id: str, amount_micros: int, kind: str, ref: str, description: str) -> bool:
    try:
        conn.execute(
            "INSERT INTO credit_ledger (org_id, amount_micros, kind, ref, description, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (org_id, amount_micros, kind, ref, description, _now()),
        )
    except sqlite3.IntegrityError:
        return False  # same (kind, ref) already recorded
    conn.execute("UPDATE orgs SET balance_micros = balance_micros + ? WHERE id = ?", (amount_micros, org_id))
    return True


def add_credits(org_id: str, amount_micros: int, kind: str, ref: str, description: str = "") -> bool:
    """Add (or remove, if negative) credits. Idempotent on (kind, ref): returns False if already applied."""
    with transaction() as conn:
        return _ledger_insert(conn, org_id, amount_micros, kind, ref, description)


def list_ledger(org_id: str, limit: int = 50, kinds: tuple[str, ...] | None = None) -> list[dict]:
    q = "SELECT * FROM credit_ledger WHERE org_id = ?"
    args: list = [org_id]
    if kinds:
        q += f" AND kind IN ({','.join('?' * len(kinds))})"
        args += list(kinds)
    with connect() as conn:
        return [dict(r) for r in conn.execute(q + " ORDER BY id DESC LIMIT ?", (*args, limit))]


# ---- usage ------------------------------------------------------------------

def _truncate_json(obj) -> str | None:
    if obj is None:
        return None
    s = obj if isinstance(obj, str) else json.dumps(obj)
    return s[:BODY_LOG_LIMIT]


def record_request(*, id: str, org_id: str, model: str, status_code: int, project_id: str | None = None,
                   api_key_id: str | None = None, source: str = "api", prompt_tokens: int = 0,
                   completion_tokens: int = 0, cost_micros: int = 0, latency_ms: int | None = None,
                   ttft_ms: int | None = None, error: str | None = None, request_body=None,
                   response_body=None) -> None:
    """Log a request and, if it cost anything, debit the org and the key — atomically."""
    with transaction() as conn:
        conn.execute(
            "INSERT INTO requests (id, org_id, project_id, api_key_id, source, model, prompt_tokens, "
            "completion_tokens, cost_micros, status_code, latency_ms, ttft_ms, error, request_body, "
            "response_body, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (id, org_id, project_id, api_key_id, source, model, prompt_tokens, completion_tokens, cost_micros,
             status_code, latency_ms, ttft_ms, error, _truncate_json(request_body), _truncate_json(response_body),
             _now()),
        )
        if cost_micros > 0:
            _ledger_insert(conn, org_id, -cost_micros, "usage", id, model)
        if api_key_id:
            conn.execute(
                "UPDATE api_keys SET spend_micros = spend_micros + ?, last_used_at = ? WHERE id = ?",
                (cost_micros, _now(), api_key_id),
            )


def list_requests(org_id: str, limit: int = 50, offset: int = 0, project_id: str | None = None) -> list[dict]:
    q = ("SELECT r.*, k.name AS api_key_name FROM requests r LEFT JOIN api_keys k ON k.id = r.api_key_id "
         "WHERE r.org_id = ?")
    args: list = [org_id]
    if project_id:
        q += " AND r.project_id = ?"
        args.append(project_id)
    with connect() as conn:
        return [dict(r) for r in conn.execute(q + " ORDER BY r.created_at DESC, r.rowid DESC LIMIT ? OFFSET ?",
                                              (*args, limit, offset))]


def get_request(org_id: str, request_id: str) -> dict | None:
    with connect() as conn:
        return _row(conn.execute("SELECT * FROM requests WHERE org_id = ? AND id = ?", (org_id, request_id)).fetchone())


def usage_by_day(org_id: str, days: int = 30) -> list[dict]:
    """One row per (day, model): requests, tokens and cost. Days are UTC 'YYYY-MM-DD'."""
    since = _now() - days * 86400
    with connect() as conn:
        rows = conn.execute(
            "SELECT date(created_at, 'unixepoch') AS day, model, COUNT(*) AS requests, "
            "SUM(prompt_tokens) AS prompt_tokens, SUM(completion_tokens) AS completion_tokens, "
            "SUM(cost_micros) AS cost_micros FROM requests WHERE org_id = ? AND created_at >= ? "
            "GROUP BY day, model ORDER BY day",
            (org_id, since),
        ).fetchall()
    return [dict(r) for r in rows]


def performance_stats(org_id: str | None = None, since_s: int = 3600) -> dict:
    """Latency percentiles and throughput over successful requests (org_id=None: all orgs)."""
    q = "SELECT latency_ms, ttft_ms, completion_tokens FROM requests WHERE status_code = 200 AND created_at >= ?"
    args: list = [_now() - since_s]
    if org_id:
        q += " AND org_id = ?"
        args.append(org_id)
    with connect() as conn:
        rows = conn.execute(q, args).fetchall()

    def pct(values: list[int], p: float) -> int | None:
        if not values:
            return None
        values = sorted(values)
        return values[min(len(values) - 1, int(p * len(values)))]

    latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    ttfts = [r["ttft_ms"] for r in rows if r["ttft_ms"] is not None]
    speeds = [r["completion_tokens"] / (r["latency_ms"] / 1000) for r in rows if r["latency_ms"]]
    return {
        "requests": len(rows),
        "latency_p50_ms": pct(latencies, 0.5), "latency_p95_ms": pct(latencies, 0.95),
        "ttft_p50_ms": pct(ttfts, 0.5), "ttft_p95_ms": pct(ttfts, 0.95),
        "tokens_per_s_avg": round(sum(speeds) / len(speeds), 1) if speeds else None,
    }
