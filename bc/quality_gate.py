"""BC dataset quality gate: rank/cull extracted episodes before training.

Input : <extract_out>/episodes.jsonl  (written by bc/extract.py)
Output: <out>/quality_report.json     per-episode verdicts + aggregate stats
        <out>/passed.jsonl            {"episode_id": ...} lines, one per
                                      pass — feed to train_bc.py --keep
        <out>/culled.jsonl            {"episode_id", "reasons"} lines

## Leaderboard snapshot schema (--leaderboard, optional)

    {
      "as_of": "2026-11-05",          # snapshot date (informational)
      "source": "kaggle leaderboard", # informational
      "ratings": {"<player name>": <float rating>, ...}
    }

Missing file/parameter => the rating gate is SKIPPED entirely (episode
episode JSONs carry only info.Agents names + +-1 rewards, no rating field,
so ratings can only come from this join).

## Rules (each behind a CLI flag)

  rating gate      both seat ratings >= --min-rating (default 1650, the
                   "top8 seed" idea: only learn from the ladder's strong
                   end; lower it to widen). --rating-rule both|any|winner.
  deck-share gate  canonical deck (sorted 60 card ids) taking more than
                   --max-deck-share (default 0.20) of all decision rows is
                   over-represented; episodes are culled lowest-quality-
                   score first until the deck's remaining share <= cap
                   (V18-style dominance control, applied to BC rows).
  mulligan gate    max seat mulligans > --max-mulligans (default 3) culls
                   (broken/degenerate setup games).
  duration gate    engine turns outside [--min-turns, --max-turns]
                   (defaults 4..30; observed ladder games run 5..20).
  sanity           status must be extract_ok, rows >= --min-rows (10),
                   rewards must match the omniscient frame result
                   (--allow-reward-mismatch to disable).

Quality score (ranking signal for the deck-share cull order; not a gate):
  1.0 - 0.25 * (leaderboard present and any seat unrated)
      - 0.10 * max(0, total mulligans - 1)
      - 0.15 * (turns > 20) - 0.30 * (turns > 25), clipped to [0, 1].

No training happens anywhere in this file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter


def deck_key(deck) -> str:
    return " ".join(str(c) for c in sorted(deck or []))


def load_episodes(extract_dir: str) -> list[dict]:
    path = os.path.join(extract_dir, "episodes.jsonl")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def gate(episodes: list[dict], args) -> tuple[list[dict], Counter]:
    lb = None
    if args.leaderboard:
        with open(args.leaderboard, encoding="utf-8") as f:
            lb = json.load(f)
    ratings = (lb or {}).get("ratings", {})
    agg = Counter()
    records = []
    for ep in episodes:
        rec = {
            "episode_id": ep.get("episode_id"),
            "day": ep.get("day"),
            "names": ep.get("names"),
            "n_rows": ep.get("n_rows", 0),
            "mulligans": ep.get("mulligans", [0, 0]),
            "turns": ep.get("turns"),
            "mirror": ep.get("mirror"),
            "score": None, "passed": False, "reasons": [],
        }
        reasons = rec["reasons"]
        if ep.get("status") != "extract_ok":
            reasons.append(f"status={ep.get('status')}")
        if (ep.get("n_rows") or 0) < args.min_rows:
            reasons.append(f"rows<{args.min_rows}")
        if ep.get("reward_consistent") is False \
                and not args.allow_reward_mismatch:
            reasons.append("reward_mismatch")
        turns = ep.get("turns")
        if turns is None or turns < args.min_turns or turns > args.max_turns:
            reasons.append(f"turns_out_of_band({turns})")
        mull = max((ep.get("mulligans") or [0, 0])[0],
                   (ep.get("mulligans") or [0, 0])[1])
        if mull > args.max_mulligans:
            reasons.append(f"mulligans>{args.max_mulligans}")

        r0 = ratings.get((ep.get("names") or [None, None])[0])
        r1 = ratings.get((ep.get("names") or [None, None])[1])
        rec["ratings"] = [r0, r1]
        rating_ok = True
        if lb is not None and args.min_rating > 0:
            have = [r for r in (r0, r1) if r is not None]
            if args.rating_rule == "both":
                rating_ok = len(have) == 2 and min(have) >= args.min_rating
            elif args.rating_rule == "any":
                rating_ok = bool(have) and max(have) >= args.min_rating
            else:   # winner
                w = 0 if (ep.get("rewards") or [0, 0])[0] == 1 else 1
                rw = (r0, r1)[w]
                rating_ok = rw is not None and rw >= args.min_rating
            if not rating_ok:
                reasons.append(
                    f"rating[{r0},{r1}]<{args.min_rating}({args.rating_rule})")
        if lb is not None and (r0 is None or r1 is None):
            rec["score"] = max(0.0, 1.0 - 0.25)
        else:
            rec["score"] = 1.0
        tot_mull = sum(ep.get("mulligans") or [0, 0])
        if tot_mull > 1:
            rec["score"] -= 0.10 * (tot_mull - 1)
        if isinstance(turns, (int, float)):
            if turns > 25:
                rec["score"] -= 0.30
            elif turns > 20:
                rec["score"] -= 0.15
        rec["score"] = round(max(0.0, min(1.0, rec["score"])), 4)
        rec["deck"] = deck_key(ep.get("decks", [None])[0] if ep.get("decks")
                               else None)
        rec["_deck1"] = deck_key(ep.get("decks", [None, None])[1]
                                 if ep.get("decks") else None)
        rec["passed"] = not reasons
        agg["total"] += 1
        agg["passed" if rec["passed"] else "culled"] += 1
        for r in reasons:
            head = r.split("(")[0].split("<")[0].split(">")[0].split("[")[0]
            agg[f"reason_{head}"] += 1
        records.append(rec)
    return records, agg


def enforce_deck_share(records: list[dict], cap: float) -> int:
    """Cull lowest-score episodes of over-represented decks until every
    deck's share of total KEPT decision rows is <= cap. Returns culled #."""
    def shares(recs):
        tot = sum(r["n_rows"] for r in recs) or 1
        per = Counter()
        for r in recs:
            if r["passed"]:
                per[r["deck"]] += r["n_rows"]
                per[r["_deck1"]] += r["n_rows"]
        return {d: n / tot for d, n in per.items()}, tot

    culled = 0
    while True:
        shares_d, tot = shares(records)
        worst_d = max(shares_d, key=lambda d: shares_d[d]) \
            if shares_d else None
        if worst_d is None or shares_d[worst_d] <= cap:
            break
        cands = [r for r in records
                 if r["passed"] and worst_d in (r["deck"], r["_deck1"])]
        if not cands:
            break
        victim = min(cands, key=lambda r: (r["score"], -r["n_rows"]))
        victim["passed"] = False
        victim["reasons"].append(
            f"deck_share>{cap} ({shares_d[worst_d]:.3f})")
        culled += 1
        if culled > len(records):      # paranoia: never loop forever
            break
    return culled


def main():
    ap = argparse.ArgumentParser(
        description="BC extraction quality gate (no training)")
    ap.add_argument("--extract-dir", required=True,
                    help="bc/extract.py output dir (has episodes.jsonl)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--leaderboard", default=None,
                    help="leaderboard snapshot JSON; omit to skip the "
                         "rating gate (schema in file header)")
    ap.add_argument("--min-rating", type=float, default=1650.0,
                    help="rating threshold (top8-style seed; default 1650)")
    ap.add_argument("--rating-rule", choices=("both", "any", "winner"),
                    default="both")
    ap.add_argument("--max-deck-share", type=float, default=0.20,
                    help="max share of decision rows per canonical deck")
    ap.add_argument("--max-mulligans", type=int, default=3)
    ap.add_argument("--min-turns", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--min-rows", type=int, default=10)
    ap.add_argument("--allow-reward-mismatch", action="store_true")
    args = ap.parse_args()

    episodes = load_episodes(args.extract_dir)
    if not episodes:
        print("no episodes found", file=sys.stderr)
        return 2
    records, agg = gate(episodes, args)
    share_culled = enforce_deck_share(records, args.max_deck_share) \
        if args.max_deck_share > 0 else 0
    agg["deck_share_culled"] = share_culled

    os.makedirs(args.out, exist_ok=True)
    keep = []
    with open(os.path.join(args.out, "quality_report.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "params": {k: v for k, v in vars(args).items()},
            "totals": dict(agg),
            "episodes": [{k: v for k, v in r.items()
                          if not k.startswith("_")} for r in records],
        }, f, indent=1)
    with open(os.path.join(args.out, "passed.jsonl"), "w",
              encoding="utf-8") as f:
        for r in records:
            if r["passed"]:
                keep.append(r["episode_id"])
                f.write(json.dumps({"episode_id": r["episode_id"],
                                    "score": r["score"]}) + "\n")
    with open(os.path.join(args.out, "culled.jsonl"), "w",
              encoding="utf-8") as f:
        for r in records:
            if not r["passed"]:
                f.write(json.dumps({"episode_id": r["episode_id"],
                                    "reasons": r["reasons"]}) + "\n")

    kept_rows = sum(r["n_rows"] for r in records if r["passed"])
    print(f"gate: {agg['passed']}/{agg['total']} passed "
          f"(kept rows {kept_rows}), culled={agg['culled']} "
          f"(deck-share culled {share_culled})")
    for k in sorted(agg):
        if k.startswith("reason_") or k == "deck_share_culled":
            print(f"  {k}: {agg[k]}")
    print(f"report -> {os.path.join(args.out, 'quality_report.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
