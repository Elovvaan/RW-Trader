#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

LIFECYCLE_RE = re.compile(r"action_id=(?P<action_id>\S+) role=(?P<role>\S+) state=(?P<state>\S+) reason_code=(?P<reason>.*)$")
KV_RE = re.compile(r"(entry|exit|qty|realized_pnl_usd|realized_pnl_pct|session_pnl|rotation_count)=(-?[0-9]+(?:\.[0-9]+)?)")


def parse_args():
    p = argparse.ArgumentParser(description="FLIP_HYPER live runtime diagnostic from RW-Trader SQLite event store")
    p.add_argument("--db", required=True, help="Path to rw-trader event sqlite DB")
    p.add_argument("--window-hours", type=float, default=6.0, help="Recent live window size in hours")
    return p.parse_args()


def parse_payload(raw):
    try:
        return json.loads(raw)
    except Exception:
        return {}


def blocked_category(reason: str) -> str:
    r = reason.lower()
    if "flip_hyper profit floor" in r or "flip_profit_floor" in r:
        return "profit_floor_blocked"
    if "score_below_threshold" in r or "compound_weak_signal" in r or "below threshold" in r:
        return "threshold_blocked"
    if "spread" in r or "slippage" in r:
        return "spread_slippage_blocked"
    if "allocation_rejected" in r or "insufficient" in r or "min_notional" in r or "notional" in r:
        return "sizing_min_notional_blocked"
    if "authority" in r or "live_requires_paper_execution" in r:
        return "authority_blocked"
    if "reconcile" in r or "stale" in r or "visibility" in r:
        return "reconcile_fill_visibility_delay"
    if "stall" in r:
        return "anti_stall_triggered"
    return "other_blocked"


def main():
    args = parse_args()
    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(hours=args.window_hours)

    rows = conn.execute(
        """
        SELECT occurred_at, event_type, payload
        FROM events
        WHERE occurred_at >= ? AND occurred_at <= ?
        ORDER BY occurred_at ASC
        """,
        (start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")),
    ).fetchall()

    counts = Counter()
    blockers = Counter()
    lifecycle_blocked = []
    flip_completed = []
    fills = []

    for row in rows:
        event_type = row["event_type"]
        payload = parse_payload(row["payload"])

        if event_type == "order_submitted":
            p = payload.get("order_submitted", payload)
            side = (p.get("side") or "").upper()
            reason = p.get("reason") or ""
            if "npc cycle=" in reason:
                if side == "BUY":
                    counts["entry_attempts"] += 1
                elif side == "SELL":
                    counts["exit_attempts"] += 1

        if event_type == "risk_check_result":
            p = payload.get("risk_check_result", payload)
            if p.get("approved"):
                side = (p.get("side") or "").upper()
                if side == "BUY":
                    counts["approved_buys"] += 1
                elif side == "SELL":
                    counts["approved_sells"] += 1

        if event_type == "order_filled":
            p = payload.get("order_filled", payload)
            side = (p.get("side") or "").upper()
            fills.append((row["occurred_at"], side, float(p.get("filled_qty") or 0.0), float(p.get("avg_fill_price") or 0.0)))

        if event_type == "operator_action":
            p = payload.get("operator_action", payload)
            action = p.get("action") or ""
            reason = p.get("reason") or ""
            if action == "npc:alpha_cycle":
                counts["total_cycles"] += 1
            elif action == "npc:lifecycle":
                m = LIFECYCLE_RE.match(reason)
                if m and m.group("state").lower() == "blocked":
                    why = m.group("reason")
                    lifecycle_blocked.append(why)
                    cat = blocked_category(why)
                    blockers[cat] += 1
                    if "sell" in why.lower() or "profit floor" in why.lower():
                        counts["blocked_exits"] += 1
                    if "buy" in why.lower() or "rebuy" in why.lower():
                        counts["blocked_rebuys"] += 1
            elif action == "npc:flip_completed":
                data = {k: float(v) for (k, v) in KV_RE.findall(reason)}
                if data:
                    flip_completed.append(data)

    counts["total_blocked"] = sum(blockers.values())
    counts["completed_flips"] = len(flip_completed)

    if flip_completed:
        session_realized = sum(x.get("realized_pnl_usd", 0.0) for x in flip_completed)
        avg_realized = session_realized / len(flip_completed)
    else:
        session_realized = 0.0
        avg_realized = 0.0

    sell_fills = [f for f in fills if f[1] == "SELL"]
    buy_fills = [f for f in fills if f[1] == "BUY"]

    rebuy_after_sell = 0
    for s in sell_fills:
        if any(b[0] > s[0] for b in buy_fills):
            rebuy_after_sell += 1

    report = {
        "window": {
            "hours": args.window_hours,
            "from_utc": start.isoformat(),
            "to_utc": end.isoformat(),
            "rows": len(rows),
        },
        "cycle_summary": {
            "total_cycles": counts["total_cycles"],
            "total_entry_attempts": counts["entry_attempts"],
            "total_approved_buys": counts["approved_buys"],
            "total_approved_sells": counts["approved_sells"],
            "total_completed_flips": counts["completed_flips"],
            "total_blocked_exits": counts["blocked_exits"],
            "total_blocked_rebuys": counts["blocked_rebuys"],
            "session_realized_pnl_usd": round(session_realized, 8),
            "avg_realized_pnl_per_flip_usd": round(avg_realized, 8),
        },
        "blocker_counts": {
            "profit_floor_blocked": blockers["profit_floor_blocked"],
            "threshold_blocked": blockers["threshold_blocked"],
            "spread_slippage_blocked": blockers["spread_slippage_blocked"],
            "sizing_min_notional_blocked": blockers["sizing_min_notional_blocked"],
            "authority_blocked": blockers["authority_blocked"],
            "reconcile_fill_visibility_delay": blockers["reconcile_fill_visibility_delay"],
            "anti_stall_triggered": blockers["anti_stall_triggered"],
            "other_blocked": blockers["other_blocked"],
        },
        "sell_rebuy_proof": {
            "sell_fills": len(sell_fills),
            "buy_fills": len(buy_fills),
            "sells_with_later_rebuy_fill": rebuy_after_sell,
            "sells_without_later_rebuy_fill": max(0, len(sell_fills) - rebuy_after_sell),
        },
        "flip_economics": flip_completed,
    }

    print(json.dumps(report, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
