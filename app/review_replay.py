"""复核结论回放 (review-conclusion replay).

Reviewed match records preserve a human conclusion: ``confirmed`` /
``overridden`` point at a chosen standard address, ``rejected`` declares the
query unmatchable. Rules and the standard address table keep evolving,
though, so a conclusion that was correct at review time may no longer match
what the matcher would do today. This module replays every reviewed record's
**original query** against the **current** standard address table with the
**default** rules (plain scoring -- the review-consensus memory is not
consulted) and compares the fresh verdict with the recorded conclusion:

* ``confirmed`` / ``overridden`` (positive decisions, target = selected_id):

  - auto-hit on the selected address   -> ``consistent`` 一致
  - auto-hit on a *different* address  -> ``conflict``   冲突
  - no auto-hit at all                 -> ``missed``     漏命中

* ``rejected`` (declared no-match):

  - any auto-hit                       -> ``false_hit``  误命中
  - no auto-hit                        -> ``consistent`` 一致

Each item carries the original conclusion (action, selected address, score
recorded at review time) next to the current best candidate and score, so
rule or address-table regressions can be located quickly. The replay is
read-only: it never writes match logs and never modifies the reviewed
records; it only re-scores the saved queries.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from . import db
from .config import Settings
from .matcher import candidate_to_dict, match_address

CATEGORY_CONSISTENT = "consistent"
CATEGORY_CONFLICT = "conflict"
CATEGORY_MISSED = "missed"
CATEGORY_FALSE_HIT = "false_hit"

VALID_CATEGORIES = (
    CATEGORY_CONSISTENT,
    CATEGORY_CONFLICT,
    CATEGORY_MISSED,
    CATEGORY_FALSE_HIT,
)

CATEGORY_LABELS = {
    CATEGORY_CONSISTENT: "一致",
    CATEGORY_CONFLICT: "冲突",
    CATEGORY_MISSED: "漏命中",
    CATEGORY_FALSE_HIT: "误命中",
}

ACTION_LABELS = {
    "confirmed": "确认首选",
    "overridden": "改派地址",
    "rejected": "判无匹配",
}

_STATUS_LABELS = {
    "auto_matched": "自动命中",
    "needs_review": "待确认",
    "no_match": "无匹配",
}

# Risks first: a conflicting auto-hit is the most urgent, a merely missed
# auto-hit the least; consistent records come last (by match id inside).
_CATEGORY_RANK = {
    CATEGORY_CONFLICT: 0,
    CATEGORY_FALSE_HIT: 1,
    CATEGORY_MISSED: 2,
    CATEGORY_CONSISTENT: 3,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _classify(row: sqlite3.Row, status: str, auto_id: int | None,
              auto_score: float | None) -> tuple[str, str]:
    """Category + human-readable reason for one replayed review record."""
    action = row["review_action"]
    selected_id = row["selected_id"]
    if action in ("confirmed", "overridden"):
        action_label = ACTION_LABELS[action]
        selected_txt = (
            f"#{selected_id}" if selected_id is not None else "已删除的地址"
        )
        if auto_id is not None:
            if selected_id is not None and auto_id == selected_id:
                return CATEGORY_CONSISTENT, (
                    f"回放自动命中 #{auto_id}，与人工{action_label}所选地址一致"
                )
            return CATEGORY_CONFLICT, (
                f"回放自动命中 #{auto_id}（{_pct(auto_score or 0.0)}），"
                f"与人工{action_label}所选地址（{selected_txt}）不同"
            )
        return CATEGORY_MISSED, (
            f"回放未自动命中（当前判定：{_STATUS_LABELS.get(status, status)}），"
            f"人工{action_label}所选地址（{selected_txt}）如今无法自动命中"
        )
    # rejected: the human declared the query unmatchable.
    if auto_id is not None:
        return CATEGORY_FALSE_HIT, (
            f"人工判定无匹配，但回放自动命中 #{auto_id}"
            f"（{_pct(auto_score or 0.0)}），规则或地址表变化可能引入了误命中"
        )
    return CATEGORY_CONSISTENT, "人工判定无匹配，回放仍未自动命中，结论一致"


def compute_replay(conn: sqlite3.Connection, settings: Settings,
                   category: str | None = None) -> dict[str, Any]:
    """Replay every reviewed record against the live table and classify it.

    ``category`` optionally restricts the returned items to one category;
    the summary counts always cover every reviewed record. The whole replay
    is recomputed on each call, so rule or address-table changes are
    reflected the next time it is opened. Nothing is persisted.
    """
    rows = conn.execute(
        """
        SELECT * FROM match_logs
         WHERE status = 'reviewed'
           AND review_action IN ('confirmed', 'overridden', 'rejected')
         ORDER BY id
        """,
    ).fetchall()
    # One snapshot of the current table is shared by every replayed query.
    addresses = db.all_addresses(conn)
    addr_by_id = {row["id"]: row for row in addresses}

    items: list[dict[str, Any]] = []
    for row in rows:
        # Pure scoring only: no review-consensus lookup, and nothing is
        # persisted -- the reviewed record keeps its saved conclusion.
        result = match_address(row["query"], addresses, settings)
        best = result.best
        auto_id = (
            best.address["id"]
            if (result.status == "auto_matched" and best is not None)
            else None
        )
        cat, reason = _classify(
            row, result.status, auto_id, best.score if best else None,
        )

        selected_id = row["selected_id"]
        selected_row = (
            addr_by_id.get(selected_id) if selected_id is not None else None
        )
        items.append({
            "match_id": row["id"],
            "query": row["query"],
            "review_action": row["review_action"],
            "review_action_label": ACTION_LABELS[row["review_action"]],
            "reviewed_at": row["reviewed_at"],
            # The score the record carried when the review was written.
            "review_score": row["score"],
            "selected_id": selected_id,
            "selected_address": (
                {"id": selected_row["id"], "full_text": selected_row["full_text"]}
                if selected_row is not None else None
            ),
            "selected_missing": selected_id is not None and selected_row is None,
            "category": cat,
            "category_label": CATEGORY_LABELS[cat],
            "replay_reason": reason,
            "replay": {
                "status": result.status,
                "reason": result.reason,
                "best_id": best.address["id"] if best else None,
                "best_score": best.score if best else None,
                "best_full_text": best.address["full_text"] if best else None,
                "candidates": [candidate_to_dict(c) for c in result.candidates],
            },
        })

    items.sort(key=lambda it: (_CATEGORY_RANK[it["category"]], it["match_id"]))

    counts = {cat: sum(1 for it in items if it["category"] == cat)
              for cat in VALID_CATEGORIES}
    summary = {
        "total_reviewed": len(items),
        **counts,
        # Everything that is not "consistent" deserves a second look.
        "risk_count": len(items) - counts[CATEGORY_CONSISTENT],
    }

    if category is not None:
        items = [it for it in items if it["category"] == category]

    return {
        "generated_at": _now(),
        **summary,
        "thresholds": {
            "auto_threshold": settings.auto_threshold,
            "margin_threshold": settings.margin_threshold,
            "reject_threshold": settings.reject_threshold,
            "token_weight": settings.token_weight,
        },
        "items": items,
    }
