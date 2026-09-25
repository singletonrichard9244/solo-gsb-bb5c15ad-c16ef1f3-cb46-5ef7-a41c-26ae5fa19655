"""Tests for the review-conclusion replay (``/api/review-replay``)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _add_addresses(client, *full_texts: str) -> None:
    r = client.post("/api/addresses", json={
        "addresses": [{"address": t} for t in full_texts],
        "skip_duplicates": True,
    })
    assert r.status_code == 200
    assert r.json()["inserted"] == len(full_texts), r.json()


def _match_save(client, query: str) -> dict:
    r = client.post("/api/match", json={"query": query, "save": True})
    assert r.status_code == 200
    data = r.json()
    assert "match_id" in data
    return data


def _review(client, match_id: int, action: str,
            selected_id: int | None = None) -> None:
    body = {"action": action}
    if selected_id is not None:
        body["selected_id"] = selected_id
    r = client.post(f"/api/matches/{match_id}/review", json=body)
    assert r.status_code == 200, r.text


def _replay(client, category: str | None = None) -> dict:
    qs = f"?category={category}" if category else ""
    r = client.get(f"/api/review-replay{qs}")
    assert r.status_code == 200
    return r.json()


def _item_for(data: dict, match_id: int) -> dict:
    matches = [it for it in data["items"] if it["match_id"] == match_id]
    assert len(matches) == 1, [it["match_id"] for it in data["items"]]
    return matches[0]


@pytest.fixture(scope="module")
def seeded(client):
    """Seed one reviewed record per replay category (unique street names)."""
    _add_addresses(
        client,
        "500 Zxreplay Ave, Replaytown 55550",
        "600 Zxreplay Ave, Replaytown 55560",
        "700 Zxreplay Ave, Replaytown 55570",
    )
    addr = {a["full_text"]: a["id"]
            for a in client.get("/api/addresses?limit=500").json()["items"]}
    a500 = addr["500 Zxreplay Ave, Replaytown 55550"]
    a600 = addr["600 Zxreplay Ave, Replaytown 55560"]
    a700 = addr["700 Zxreplay Ave, Replaytown 55570"]

    # consistent: auto-matched query, human confirms the auto-hit address.
    m = _match_save(client, "500 Zxreplay Ave, Replaytown 55550")
    assert m["status"] == "auto_matched" and m["best_id"] == a500
    _review(client, m["match_id"], "confirmed")
    consistent_id = m["match_id"]

    # conflict: human overrode to #600, but the replay auto-hits #500.
    m = _match_save(client, "500 Zxreplay Ave, Replaytown 55550")
    _review(client, m["match_id"], "overridden", selected_id=a600)
    conflict_id = m["match_id"]

    # missed: house-number conflict keeps the replay below the auto
    # threshold, so the human-confirmed address is not auto-hit any more.
    m = _match_save(client, "701 Zxreplay Ave, Replaytown 55570")
    assert m["status"] == "needs_review" and m["best_id"] == a700
    _review(client, m["match_id"], "confirmed")
    missed_id = m["match_id"]

    # false_hit: human declared no-match, yet the replay auto-hits #600.
    m = _match_save(client, "600 Zxreplay Ave, Replaytown 55560")
    assert m["status"] == "auto_matched" and m["best_id"] == a600
    _review(client, m["match_id"], "rejected")
    false_hit_id = m["match_id"]

    # consistent (rejected): human declared no-match and the replay agrees.
    m = _match_save(client, "1 Zxnohit Way, Nowhereland 00000")
    assert m["status"] == "no_match"
    _review(client, m["match_id"], "rejected")
    rejected_consistent_id = m["match_id"]

    return {
        "a500": a500, "a600": a600, "a700": a700,
        "consistent": consistent_id,
        "conflict": conflict_id,
        "missed": missed_id,
        "false_hit": false_hit_id,
        "rejected_consistent": rejected_consistent_id,
    }


def test_replay_classifies_each_category(client, seeded):
    data = _replay(client)

    consistent = _item_for(data, seeded["consistent"])
    assert consistent["category"] == "consistent"
    assert consistent["category_label"] == "一致"
    assert consistent["review_action"] == "confirmed"
    assert consistent["review_action_label"] == "确认首选"
    assert consistent["selected_id"] == seeded["a500"]
    assert consistent["selected_address"]["full_text"] == \
        "500 Zxreplay Ave, Replaytown 55550"
    assert consistent["selected_missing"] is False
    assert consistent["replay"]["status"] == "auto_matched"
    assert consistent["replay"]["best_id"] == seeded["a500"]
    assert consistent["replay"]["best_score"] >= 0.8

    conflict = _item_for(data, seeded["conflict"])
    assert conflict["category"] == "conflict"
    assert conflict["category_label"] == "冲突"
    assert conflict["review_action"] == "overridden"
    assert conflict["selected_id"] == seeded["a600"]
    assert conflict["replay"]["status"] == "auto_matched"
    assert conflict["replay"]["best_id"] == seeded["a500"]
    assert conflict["replay"]["best_full_text"] == \
        "500 Zxreplay Ave, Replaytown 55550"

    missed = _item_for(data, seeded["missed"])
    assert missed["category"] == "missed"
    assert missed["category_label"] == "漏命中"
    assert missed["selected_id"] == seeded["a700"]
    assert missed["replay"]["status"] == "needs_review"
    # the human-chosen address is still the current best candidate
    assert missed["replay"]["best_id"] == seeded["a700"]

    false_hit = _item_for(data, seeded["false_hit"])
    assert false_hit["category"] == "false_hit"
    assert false_hit["category_label"] == "误命中"
    assert false_hit["review_action"] == "rejected"
    assert false_hit["selected_id"] is None
    assert false_hit["selected_address"] is None
    assert false_hit["replay"]["status"] == "auto_matched"
    assert false_hit["replay"]["best_id"] == seeded["a600"]

    rejected_ok = _item_for(data, seeded["rejected_consistent"])
    assert rejected_ok["category"] == "consistent"
    assert rejected_ok["review_action"] == "rejected"
    assert rejected_ok["replay"]["status"] == "no_match"
    assert rejected_ok["replay"]["best_id"] is None or \
        rejected_ok["replay"]["best_score"] < 0.35


def test_replay_summary_counts_and_risk_count(client, seeded):
    data = _replay(client)
    ours = [seeded[k] for k in
            ("consistent", "conflict", "missed", "false_hit",
             "rejected_consistent")]
    assert data["total_reviewed"] >= len(ours)
    assert data["consistent"] >= 2
    assert data["conflict"] >= 1
    assert data["missed"] >= 1
    assert data["false_hit"] >= 1
    assert data["risk_count"] == (
        data["conflict"] + data["missed"] + data["false_hit"]
    )
    assert data["total_reviewed"] == (
        data["consistent"] + data["risk_count"]
    )
    assert data["thresholds"]["auto_threshold"] == pytest.approx(0.80)
    assert data["generated_at"]
    # risks sort before consistent records
    ranks = {"conflict": 0, "false_hit": 1, "missed": 2, "consistent": 3}
    order = [ranks[it["category"]] for it in data["items"]]
    assert order == sorted(order)


def test_replay_category_filter(client, seeded):
    for cat, expected_id in (
        ("conflict", seeded["conflict"]),
        ("missed", seeded["missed"]),
        ("false_hit", seeded["false_hit"]),
    ):
        data = _replay(client, cat)
        assert data["items"], cat
        assert all(it["category"] == cat for it in data["items"])
        _item_for(data, expected_id)

    data = _replay(client, "consistent")
    assert all(it["category"] == "consistent" for it in data["items"])
    _item_for(data, seeded["consistent"])
    _item_for(data, seeded["rejected_consistent"])

    # summary counts are independent of the filter
    full = _replay(client)
    filtered = _replay(client, "conflict")
    for key in ("total_reviewed", "consistent", "conflict", "missed",
                "false_hit", "risk_count"):
        assert filtered[key] == full[key]


def test_replay_is_readonly(client, seeded):
    before_total = client.get("/api/matches?limit=1").json()["total"]
    before_items = {
        it["match_id"]: it for it in _replay(client)["items"]
        if it["match_id"] in set(seeded.values())
    }
    before_logs = {
        mid: client.get(f"/api/matches/{mid}").json()
        for mid in before_items
    }

    _replay(client)
    _replay(client, "conflict")

    assert client.get("/api/matches?limit=1").json()["total"] == before_total
    for mid, log in before_logs.items():
        after = client.get(f"/api/matches/{mid}").json()
        assert after["status"] == "reviewed"
        assert after["review_action"] == log["review_action"]
        assert after["selected_id"] == log["selected_id"]
        assert after["reviewed_at"] == log["reviewed_at"]
        assert after["score"] == log["score"]


def test_replay_invalid_category(client):
    r = client.get("/api/review-replay", params={"category": "bogus"})
    assert r.status_code == 400


def test_replay_empty_on_fresh_database():
    # A throwaway database with only the sample addresses (and no reviewed
    # records at all) yields an empty replay.
    import os
    import tempfile
    from pathlib import Path

    from app import db, review_replay
    from app import ingest as ingest_mod
    from app.config import get_settings

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    old = os.environ.get("ADDRESS_DB_PATH")
    os.environ["ADDRESS_DB_PATH"] = path
    conn = None
    try:
        conn = db.connect()
        db.init_db(conn)
        sample = Path(get_settings().sample_csv)
        ingest_mod.add_many(conn, ingest_mod.parse_csv(sample.read_bytes()))
        out = review_replay.compute_replay(conn, get_settings())
        assert out["items"] == []
        assert out["total_reviewed"] == 0
        assert out["consistent"] == 0 and out["conflict"] == 0
        assert out["missed"] == 0 and out["false_hit"] == 0
        assert out["risk_count"] == 0
    finally:
        if conn is not None:
            conn.close()
        if old is None:
            os.environ.pop("ADDRESS_DB_PATH", None)
        else:
            os.environ["ADDRESS_DB_PATH"] = old
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(path + suffix)
            except OSError:
                pass
