"""Tests for the platform database layer (minilab.db)."""

import pytest

from minilab import db


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    db.configure(tmp_path / "test.db")
    db.init_db()
    yield
    db.configure(None)


def make_org():
    user = db.create_user("Alice@Example.com", "correct horse", "Alice")
    org = db.create_org("Acme", user["id"], signup_credit_usd=1.0)
    project = db.list_projects(org["id"])[0]
    return user, org, project


def test_users_and_sessions():
    user = db.create_user("Alice@Example.com", "correct horse", "Alice")
    with pytest.raises(ValueError):
        db.create_user("alice@example.com", "other", "")
    assert db.authenticate("ALICE@example.com", "correct horse")["id"] == user["id"]
    assert db.authenticate("alice@example.com", "wrong") is None
    token = db.create_session(user["id"])
    assert db.get_user_by_session(token)["email"] == "alice@example.com"
    db.delete_session(token)
    assert db.get_user_by_session(token) is None


def test_org_gets_default_project_and_signup_credit():
    user, org, project = make_org()
    assert project["name"] == "Default project"
    assert org["balance_micros"] == 1_000_000
    assert db.is_member(org["id"], user["id"])
    assert [o["id"] for o in db.list_user_orgs(user["id"])] == [org["id"]]


def test_api_keys_are_hashed_and_revocable():
    user, org, project = make_org()
    row, secret = db.create_api_key(org["id"], project["id"], "ci", user["id"], spend_limit_usd=0.5)
    assert secret.startswith("sk-mini-") and "key_hash" not in row
    key = db.lookup_api_key(secret)
    assert key["id"] == row["id"] and key["spend_limit_micros"] == 500_000
    assert db.lookup_api_key(secret + "x") is None and db.lookup_api_key("sk-other") is None
    assert not db.revoke_api_key("org_other", row["id"])  # other orgs can't revoke it
    assert db.revoke_api_key(org["id"], row["id"])
    assert db.lookup_api_key(secret) is None


def test_usage_debits_org_and_key_atomically():
    user, org, project = make_org()
    row, secret = db.create_api_key(org["id"], project["id"], "ci", user["id"])
    db.record_request(id="chatcmpl-1", org_id=org["id"], project_id=project["id"], api_key_id=row["id"],
                      model="prelude-1", status_code=200, prompt_tokens=100, completion_tokens=50,
                      cost_micros=125, latency_ms=200, ttft_ms=20)
    db.record_request(id="chatcmpl-2", org_id=org["id"], model="prelude-1", status_code=500, error="boom")
    assert db.get_balance_micros(org["id"]) == 1_000_000 - 125
    assert db.lookup_api_key(secret)["spend_micros"] == 125
    assert [r["id"] for r in db.list_requests(org["id"])] == ["chatcmpl-2", "chatcmpl-1"]
    [day] = db.usage_by_day(org["id"])
    assert (day["requests"], day["cost_micros"]) == (2, 125)
    stats = db.performance_stats(org["id"])
    assert stats["requests"] == 1 and stats["latency_p50_ms"] == 200


def test_credits_are_idempotent():
    _, org, _ = make_org()
    assert db.add_credits(org["id"], 5_000_000, "purchase", "cs_test_1")
    assert not db.add_credits(org["id"], 5_000_000, "purchase", "cs_test_1")  # replayed webhook
    assert db.get_balance_micros(org["id"]) == 6_000_000
    assert db.format_usd(db.get_balance_micros(org["id"])) == "$6.00"
