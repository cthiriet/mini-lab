-- mini-lab platform database (SQLite).
-- Money is stored as integer micro-dollars (1 USD = 1_000_000 micros).

CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,           -- user_...
    email         TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL DEFAULT '',
    password_hash TEXT NOT NULL,
    created_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,              -- sha256 of the cookie value
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS orgs (
    id             TEXT PRIMARY KEY,          -- org_...
    name           TEXT NOT NULL,
    balance_micros INTEGER NOT NULL DEFAULT 0, -- cached sum of credit_ledger
    created_at     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS org_members (
    org_id  TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role    TEXT NOT NULL DEFAULT 'owner',    -- owner | member
    PRIMARY KEY (org_id, user_id)
);

CREATE TABLE IF NOT EXISTS projects (
    id         TEXT PRIMARY KEY,              -- proj_...
    org_id     TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    id                 TEXT PRIMARY KEY,      -- key_...
    org_id             TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    project_id         TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name               TEXT NOT NULL,
    key_hash           TEXT NOT NULL UNIQUE,  -- sha256 of the secret; the secret itself is never stored
    key_hint           TEXT NOT NULL,         -- "sk-mini-abcd...wxyz" for display
    created_by         TEXT REFERENCES users(id) ON DELETE SET NULL,
    created_at         INTEGER NOT NULL,
    last_used_at       INTEGER,
    revoked_at         INTEGER,
    spend_micros       INTEGER NOT NULL DEFAULT 0,
    spend_limit_micros INTEGER,               -- NULL = no limit
    rpm_limit          INTEGER,               -- NULL = default from settings
    tpm_limit          INTEGER
);

CREATE TABLE IF NOT EXISTS credit_ledger (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id        TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    amount_micros INTEGER NOT NULL,           -- positive = credit, negative = usage
    kind          TEXT NOT NULL,              -- grant | purchase | usage | refund
    ref           TEXT NOT NULL,              -- stripe session id, request id, ...
    description   TEXT NOT NULL DEFAULT '',
    created_at    INTEGER NOT NULL,
    UNIQUE (kind, ref)                        -- makes webhooks and retries idempotent
);

CREATE TABLE IF NOT EXISTS requests (
    id                TEXT PRIMARY KEY,       -- chatcmpl-...
    org_id            TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    project_id        TEXT,
    api_key_id        TEXT,                   -- NULL for first-party traffic (playground, chat)
    source            TEXT NOT NULL DEFAULT 'api', -- api | playground | chat
    model             TEXT NOT NULL,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_micros       INTEGER NOT NULL DEFAULT 0,
    status_code       INTEGER NOT NULL,
    latency_ms        INTEGER,
    ttft_ms           INTEGER,                -- time to first token (streaming)
    error             TEXT,
    request_body      TEXT,                   -- JSON, truncated
    response_body     TEXT,                   -- JSON, truncated
    created_at        INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_requests_org_time ON requests(org_id, created_at);
CREATE INDEX IF NOT EXISTS idx_ledger_org_time ON credit_ledger(org_id, created_at);
