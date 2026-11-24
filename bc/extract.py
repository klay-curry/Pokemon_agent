"""Kaggle official episode JSON -> BC decision rows (top2 encoder path).

Data source (2026-11-16 coordinator update): pre-extracted episode JSONs at
/root/autodl-fs/pokemon_RL_old_bc/bc_extract/<YYYY-MM-DD>/<episode_id>.json
(READ-ONLY; same schema as the official episode zips).  Pass the day dir via
--input.  The original plan (stream single members out of the ~700 MB daily
episode zips with zipfile) is no longer needed but stays trivially possible:
`--zip ARCHIVE --limit N` streams members without ever extracting the archive.

## Why we encode the RECORDED observations instead of re-simulating

A BattleHandle replay (BattleStart(recorded decks) + feed recorded actions)
was built and probed first (BUILD_LOG Step 13).  Finding: the engine's
opening deal is NOT a deterministic function of the two decklists — after
the first-player question the replayed game's hand diverges from the
recording (probe: recorded hand[1].id 86 vs replayed 1406 at the same
engine step).  The recorded actions therefore cannot be re-fed into a fresh
engine battle state-by-state, and no obs-level parity with a re-simulation
is possible.

But this does not block BC extraction, because the recording itself IS the
engine's own output: every agent entry's `observation` holds exactly the
GetBattleData document (keys `current`, `logs`, `select`; runner extras
`step`/`remainingOverageTime`/`search_begin_input` are ignored by the
encoder).  So we walk the recording the way the env would have walked the
live game and feed the same objects to the same code:

  InfoTracker(seat deck).update(obs)   — fresh obs, per seat
  OracleLedger(decks).on_seat_obs(obs) — prize truth seeded from the
      omniscient `visualize` frames embedded at steps[0][0] (the env's
      handle.visualize() equivalent; we pin the first frame with
      turn >= 1 and both prize lists face-up, which is the same state the
      env freezes on, then freeze the ledger exactly like OracleLedger)
  encode(row, obs, picked, stop_ok, forced_run, tracker) + write_oracle

## Episode format mapping (kaggle-environments cabt, module_version 1.32.7)

  steps[k][i].observation      obs the engine posed seat i at step k
                               (select=None & current=None only at step 0
                               = the deck call)
  steps[k][i].action           the answer seat i produced at step k-1;
                               steps[1][i].action = the 60-card deck
  steps[k][i].status           'ACTIVE' = seat i was asked at step k
                               (INACTIVE entries carry stale filler actions,
                               filtered by requiring ACTIVE + legality)
  steps[0][0]['visualize']     full omniscient replay frames (one per
                               engine decision, NOT one per step; frame
                               count == steps count +1 on some episodes);
                               frame[-1].current.result cross-checks rewards
  info.Agents / rewards        player names, +-1 match result (BO3 config
                               key exists but every sampled episode holds
                               exactly one game — deck_calls==0 at k>0 in
                               24/24 samples; the walker still supports
                               mid-episode deck calls defensively)

Alignment contract used here (validated on 24 sampled episodes across all
4 days, 0 illegal pairings): for k>=1, steps[k][i].action answers
steps[k-1][i].observation iff steps[k-1][i].status == 'ACTIVE' and the
action is legal for that observation (index range, min/max count, no dups).

## Row semantics — identical to ptcg/rl/env.py training rows

  * forced selects (forced_answer() in ptcg/rl/env.py) produce NO row;
    per-seat count feeds the row's FORCED_RUN feature (reset per decision).
  * multi-selects decompose into sequential single-pick rows with
    picked-prefix state, then one STOP row (label STOP=63) iff the recorded
    answer was shorter than maxCount (voluntary stop; maxCount submissions
    are auto-terminated exactly like env._step_game).
  * rows_per_decision = len(answer) + (1 if len < maxCount else 0).
  * labels are CLASS-REPRESENTATIVE indices under the encoder's
    equivalence-class dedup (ptcg/rl/encoder.py keeps the first option of
    each class in opt_mask).  We mirror the class-key computation and
    cross-validate it on every row: mirrored_rep(j)==j <=> mask[j]==1 for
    all non-picked j (violations abort the run — never silently mislabel).

## Output (dir given by --out)

  <epid>.npz        rows float32 (n, OBS_SIZE=10944), labels int16 (n,),
                    seats int8 (n,)
  episodes.jsonl    one metadata line per episode (players, decks, rewards,
                    n_steps/questions/forced/decisions/rows, mulligans,
                    turns, result, quality-relevant stats, row counts)
  manifest.json     run summary + schema notes (written after --self-check)

Usage:
  python3 bc/extract.py --input /root/autodl-fs/pokemon_RL_old_bc/bc_extract/2026-11-05 \
      --out /root/autodl-tmp/bc_out/2026-11-05 --limit 3 --self-check

No training happens anywhere in this file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "data"))

import numpy as np

from ptcg.rl.buffers import OBS_SIZE, MAX_OPTIONS, STOP, OFFSETS
from ptcg.rl.encoder import encode, write_oracle
from ptcg.rl.env import forced_answer
from ptcg.rl.oracle import OracleLedger
from ptcg.tracker import InfoTracker

# ---- AreaType / OptionType values (mirror data/cg/api.py, as encoder.py) ----
DECK, HAND, DISCARD, ACTIVE, BENCH, PRIZE, STADIUM_A, ENERGY_A, TOOL_A = \
    1, 2, 3, 4, 5, 6, 7, 8, 9
LOOKING_A = 12
OT_NUMBER, OT_YES, OT_NO, OT_CARD, OT_TOOL_CARD, OT_ENERGY_CARD, OT_ENERGY, \
    OT_PLAY, OT_ATTACH, OT_EVOLVE, OT_ABILITY, OT_DISCARD, OT_RETREAT, \
    OT_ATTACK, OT_END, OT_SKILL, OT_SPECIAL = range(17)
_CARD_ZONES = {DECK, HAND, DISCARD, PRIZE, LOOKING_A}


class ExtractError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# encoder class-key mirror (cross-validated against the written opt_mask
# on every row; see rep_crosscheck below).  resolve_card/_key_option are
# verbatim ports of ptcg/rl/encoder.py's internals — do not "improve" them.
# --------------------------------------------------------------------------
def _resolve_card(cur: dict, sel: dict, pidx, area, idx):
    try:
        if area == HAND:
            cards = cur["players"][pidx]["hand"]
        elif area == DISCARD:
            cards = cur["players"][pidx]["discard"]
        elif area == PRIZE:
            cards = cur["players"][pidx]["prize"]
        elif area == DECK:
            cards = sel.get("deck")
        elif area == LOOKING_A:
            cards = cur.get("looking")
        elif area == ACTIVE:
            c = cur["players"][pidx]["active"][idx]
            return 0 if c is None else c["id"]
        elif area == BENCH:
            return cur["players"][pidx]["bench"][idx]["id"]
        elif area == STADIUM_A:
            return cur["stadium"][idx]["id"]
        else:
            return 0
        c = cards[idx] if cards else None
        return 0 if c is None else c["id"]
    except (IndexError, TypeError, KeyError):
        return 0


def _opt_card_id(cur, sel, seat, o):
    ot = o.get("type", 0)
    area = o.get("area")
    idx = o.get("index")
    pidx = o.get("playerIndex")
    card_id = o.get("cardId") or 0
    if ot == OT_PLAY:
        card_id = _resolve_card(cur, sel, seat, HAND, idx)
    elif ot in (OT_CARD, OT_DISCARD, OT_ABILITY):
        if not card_id:
            card_id = _resolve_card(cur, sel, pidx if pidx is not None
                                    else seat, area, idx)
    elif ot in (OT_ATTACH, OT_EVOLVE):
        card_id = _resolve_card(cur, sel, seat, area, idx)
    return card_id


def _opt_key(cur, sel, seat, o):
    """Equivalence-class key of one option dict — mirror of encoder.encode."""
    ot = o.get("type", 0)
    area = o.get("area")
    idx = o.get("index")
    pidx = o.get("playerIndex")
    card_id = _opt_card_id(cur, sel, seat, o)
    if ot == OT_SKILL:
        return (ot, card_id, o.get("serial"))
    board_target = (area in (ACTIVE, BENCH)) or o.get("inPlayArea") is not None
    if board_target or ot in (OT_TOOL_CARD, OT_ENERGY_CARD, OT_ENERGY):
        return (ot, area, idx, pidx, o.get("inPlayArea"),
                o.get("inPlayIndex"), card_id, o.get("count"),
                _resolve_card(cur, sel, pidx if pidx is not None else seat,
                              area, idx)
                if ot in (OT_TOOL_CARD, OT_ENERGY_CARD, OT_ENERGY) else None)
    if area in _CARD_ZONES or ot == OT_PLAY:
        return (ot, area, pidx, card_id)
    return (ot, o.get("number"), o.get("attackId"),
            o.get("specialConditionType"))


def option_reps(obs: dict, picked: list[int]) -> dict[int, int]:
    """option index -> kept class-representative index, under the same
    first-occurrence-per-class rule as encoder.encode (picked indices
    excluded from registration, exactly as there)."""
    cur = obs["current"]
    sel = obs["select"]
    seat = cur["yourIndex"]
    options = sel["option"]
    n = min(len(options), MAX_OPTIONS - 1)
    picked_set = set(picked)
    classes: dict = {}
    reps: dict[int, int] = {}
    for j in range(n):
        if j in picked_set:
            continue                     # encoder `continue`s before register
        key = _opt_key(cur, sel, seat, options[j])
        if key not in classes:
            classes[key] = j
        reps[j] = classes[key]
    return reps


def answer_label(obs: dict, picked: list[int], target: int) -> int:
    """Class-representative label for the pick `target` (an option index)."""
    reps = option_reps(obs, picked)
    if target not in reps:
        raise ExtractError(
            f"pick {target} outside encodable options "
            f"(n={len(obs['select']['option'])})")
    return reps[target]


def rep_crosscheck(obs: dict, picked: list[int], row: np.ndarray) -> int:
    """ mirrored rep(j)==j  <=>  mask[j]==1, for all non-picked j in range.
    Returns violation count (must stay 0)."""
    a = OFFSETS["opt_mask"][0]
    mask = row[a:a + MAX_OPTIONS]
    reps = option_reps(obs, picked)
    sel = obs["select"]
    n = min(len(sel["option"]), MAX_OPTIONS - 1)
    picked_set = set(picked)
    bad = 0
    for j in range(n):
        is_rep = bool(mask[j] != 0.0)
        mine = (j in reps and reps[j] == j)
        if is_rep != mine:
            bad += 1
    # picked slots must never be legal
    for j in picked_set:
        if j < n and mask[j] != 0.0:
            bad += 1
    return bad


OFFSETS = OFFSETS   # re-exported for readability; imported from buffers


# --------------------------------------------------------------------------
# episode walking
# --------------------------------------------------------------------------
def _legal(obs: dict, act) -> bool:
    sel = obs.get("select")
    if sel is None:
        return isinstance(act, list) and len(act) == 60
    if not isinstance(act, list):
        return False
    opts = sel.get("option") or []
    mn = sel.get("minCount", 0)
    mx = sel.get("maxCount", 10 ** 9)
    return (all(isinstance(a, int) and 0 <= a < len(opts) for a in act)
            and mn <= len(act) <= mx and len(set(act)) == len(act))


def walk_questions(steps: list):
    """Yield (k, seat, question_obs, answer) for every real engine question.

    k>=1; the question sits at steps[k-1][seat] (ACTIVE gate), its answer at
    steps[k][seat].action.  Deck calls (select None, current None) come back
    as answer length 60.  Returns (generator, counters dict filled on the
    fly by the consumer — counters are maintained by extract_episode)."""
    for k in range(1, len(steps)):
        for seat in (0, 1):
            prev = steps[k - 1][seat]
            if prev.get("status") != "ACTIVE":
                continue
            obs = prev.get("observation") or {}
            if obs.get("current") is None and obs.get("select") is None:
                # deck call (step 0 only, in practice)
                yield k, seat, obs, steps[k][seat].get("action")
                continue
            if obs.get("current") is None:
                continue
            act = steps[k][seat].get("action")
            yield k, seat, obs, act


def pin_prizes(frames) -> tuple[Counter, Counter] | None:
    """First omniscient frame with turn >= 1 and both prize lists face-up:
    the post-setup prize multiset, i.e. the state OracleLedger freezes on
    in the env.  None if the recording has no such frame."""
    for f in frames or []:
        cur = f.get("current") or {}
        if cur.get("turn", 0) is None or cur.get("turn", 0) < 1:
            continue
        pls = cur.get("players") or []
        if len(pls) != 2:
            continue
        out = []
        ok = True
        for s in (0, 1):
            pz = pls[s].get("prize") or []
            ids = [c["id"] for c in pz if c is not None and "id" in c]
            if len(ids) != len(pz) or not ids:
                ok = False
                break
            out.append(Counter(ids))
        if ok:
            return out[0], out[1]
    return None


def count_mulligans(obs: dict, seat: int, last_sig) -> tuple[int, object]:
    """HAS_BASIC_POKEMON(hasBasicPokemon=False) events for `seat`, deduped
    against the seat's previous obs window (the engine repeats the same
    logs across consecutive questions to the same seat)."""
    lg = obs.get("logs") or []
    sig = json.dumps(lg, sort_keys=True)
    n = 0
    if sig != last_sig:
        for e in lg:
            if (e.get("type") == 1 and e.get("hasBasicPokemon") is False
                    and e.get("playerIndex") == seat):
                n += 1
    return n, sig


def extract_episode(path: str, episode_id: str, day: str) -> dict:
    """Extract one episode.  Returns the episodes.jsonl metadata dict
    (meta['status'] in {extract_ok, no_frames, ...}); rows/labels/seats are
    returned in meta['_arrays'] for the caller to persist."""
    with open(path, encoding="utf-8") as f:
        ep = json.load(f)
    steps = ep["steps"]
    names = [a.get("Name") for a in (ep.get("info", {}).get("Agents") or [{},
            {}])]
    rewards = ep.get("rewards")

    meta = {
        "episode_id": episode_id, "day": day, "file": os.path.basename(path),
        "names": names, "rewards": rewards,
        "module_version": ep.get("module_version"),
        "n_steps": len(steps),
        "decks": None, "mirror": None,
        "n_questions": 0, "n_deck_calls": 0, "n_forced": 0,
        "n_decisions": 0, "n_rows": 0, "n_stop_rows": 0,
        "voluntary_stops": 0, "n_mismatch_answers": 0,
        "rep_violations": 0, "mask_violations": 0, "nonfinite_rows": 0,
        "overflow_options": 0, "option_cap_events": 0,
        "mulligans": [0, 0], "turns": None, "result": None,
        "frame_result": None, "reward_consistent": None,
        "prize_frames_found": None,
        "status": "extract_ok", "reason": None,
    }

    frames = None
    for seat in (0, 1):
        v = steps[0][seat].get("visualize") if steps[0][seat] else None
        if isinstance(v, list) and v:
            frames = v
            break
    if frames is None:
        meta["status"] = "no_frames"
        meta["reason"] = "no visualize frames at step 0"
        return meta
    meta["frame_result"] = (frames[-1].get("current") or {}).get("result")

    decks = [None, None]
    trackers = None
    ledger = None
    rows: list[np.ndarray] = []
    labels: list[int] = []
    seats: list[int] = []
    last_log_sig = [None, None]
    forced_run = [0, 0]
    scratch = np.zeros(OBS_SIZE, dtype=np.float32)

    def start_game(d0, d1):
        nonlocal trackers, ledger
        trackers = [InfoTracker(d0), InfoTracker(d1)]
        ledger = OracleLedger((d0, d1), None)
        pin = pin_prizes(frames)
        meta["prize_frames_found"] = pin is not None
        if pin is not None:
            ledger.prize[0], ledger.prize[1] = pin
            ledger._frozen = True   # same freeze the env reaches at turn 1
        else:
            # degraded: oracle prize split stays unknown (emit() still finite)
            ledger._frozen = True

    def emit_row(obs, picked, stop_ok, seat):
        nonlocal scratch
        encode(scratch, obs, picked, stop_ok, forced_run[seat],
               tracker=trackers[seat])
        write_oracle(scratch, *ledger.emit(obs))
        rows.append(scratch.copy())
        seats.append(seat)
        meta["n_rows"] += 1
        if not np.isfinite(scratch).all():
            meta["nonfinite_rows"] += 1
        forced_run[seat] = 0    # env resets the run at the encoded row

    for k, seat, obs, act in walk_questions(steps):
        cur = obs.get("current")
        # -- deck call: (re)start the game -------------------------------
        if cur is None:
            if not (isinstance(act, list) and len(act) == 60):
                meta["n_mismatch_answers"] += 1
                continue
            if decks[0] is not None and decks[1] is not None:
                decks = [None, None]     # defensive: next game of a BO3 match
            decks[seat] = list(act)
            meta["n_deck_calls"] += 1
            if decks[0] is not None and decks[1] is not None:
                start_game(decks[0], decks[1])
                meta["decks"] = [list(decks[0]), list(decks[1])]
                meta["mirror"] = sorted(decks[0]) == sorted(decks[1])
            continue
        # -- normal question --------------------------------------------
        if trackers is None:
            meta["status"] = "no_decks"
            meta["reason"] = f"game question before both decks at step {k}"
            return meta
        # env `_ingest`: every fresh engine obs feeds the asked seat's
        # tracker and the oracle ledger exactly once (forced or not)
        trackers[seat].update(obs)
        ledger.on_seat_obs(obs)
        if cur.get("result") != -1:
            continue            # terminal obs carries no answer
        m, sig = count_mulligans(obs, seat, last_log_sig[seat])
        meta["mulligans"][seat] += m
        last_log_sig[seat] = sig
        meta["n_questions"] += 1
        sel = obs["select"]
        if sel is None or not _legal(obs, act):
            meta["n_mismatch_answers"] += 1
            continue
        fa = forced_answer(sel)
        if fa is not None:
            meta["n_forced"] += 1
            forced_run[seat] += 1
            if sorted(act) != sorted(fa):
                meta["n_mismatch_answers"] += 1
            continue
        # -- real decision: env-decomposed rows --------------------------
        meta["n_decisions"] += 1
        mn, mx = sel["minCount"], sel["maxCount"]
        if len(sel["option"]) > MAX_OPTIONS - 1:
            meta["option_cap_events"] += 1
        ans = list(dict.fromkeys(act))       # engine bans dups; dedupe safe
        for j, tgt in enumerate(ans):
            if tgt >= MAX_OPTIONS - 1:
                meta["overflow_options"] += 1
                continue
            picked = ans[:j]
            stop_ok = (len(picked) >= mn and (mx > 1 or mn == 0))
            emit_row(obs, picked, stop_ok, seat)
            try:
                lab = answer_label(obs, picked, tgt)
            except ExtractError:
                meta["mask_violations"] += 1
                labels.append(-1)
                continue
            labels.append(lab)
            a0 = OFFSETS["opt_mask"][0]
            if lab < 0 or lab >= MAX_OPTIONS or \
                    scratch[a0 + lab] == 0.0:
                meta["mask_violations"] += 1
            # class-mirror crosscheck (cheap: n <= 63 keys)
            meta["rep_violations"] += rep_crosscheck(obs, picked, rows[-1])
        if len(ans) < mx:
            # voluntary stop: submit before reaching maxCount
            meta["voluntary_stops"] += 1
            stop_ok = (len(ans) >= mn and (mx > 1 or mn == 0))
            emit_row(obs, ans, stop_ok, seat)
            labels.append(STOP)
            meta["n_stop_rows"] += 1
            a0 = OFFSETS["opt_mask"][0]
            if stop_ok and rows[-1][a0 + STOP] == 0.0:
                meta["mask_violations"] += 1
            meta["rep_violations"] += rep_crosscheck(obs, ans, rows[-1])

    if decks[0] is None or decks[1] is None:
        meta["status"] = "no_decks"
        meta["reason"] = "never received both deck calls"
        return meta
    meta["turns"] = (frames[-1].get("current") or {}).get("turn")
    res = meta["frame_result"]
    if rewards in ([1, -1], [-1, 1]) and res in (0, 1):
        meta["reward_consistent"] = (
            (rewards[0] == 1 and res == 0)
            or (rewards[1] == 1 and res == 1))
    else:
        meta["reward_consistent"] = rewards == [0, 0] and res == 2
    meta["_arrays"] = (
        np.asarray(rows, dtype=np.float32) if rows else
        np.zeros((0, OBS_SIZE), dtype=np.float32),
        np.asarray(labels, dtype=np.int16) if labels else
        np.zeros((0,), dtype=np.int16),
        np.asarray(seats, dtype=np.int8) if seats else
        np.zeros((0,), dtype=np.int8))
    return meta


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="episode JSON dir -> BC decision rows (no training)")
    ap.add_argument("--input", required=True,
                    help="day directory of extracted episode JSONs "
                         "(read-only, e.g. .../bc_extract/2026-11-05)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N episodes (0 = all)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-extract episodes whose npz already exists")
    ap.add_argument("--self-check", action="store_true",
                    help="print aggregate stats and fail on violations")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = sorted(f for f in os.listdir(args.input) if f.endswith(".json"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"no episode JSONs in {args.input}", file=sys.stderr)
        return 2

    meta_path = os.path.join(args.out, "episodes.jsonl")
    agg = Counter()
    t0 = time.time()
    with open(meta_path, "a", encoding="utf-8") as mf:
        for name in files:
            episode_id = name[:-5]
            npz_path = os.path.join(args.out, f"{episode_id}.npz")
            if os.path.exists(npz_path) and not args.overwrite:
                agg["skipped_existing"] += 1
                continue
            path = os.path.join(args.input, name)
            try:
                meta = extract_episode(path, episode_id,
                                       os.path.basename(args.input.rstrip("/")))
            except Exception as e:      # one bad episode must not stop a day
                agg["episodes_crashed"] += 1
                mf.write(json.dumps({
                    "episode_id": episode_id,
                    "day": os.path.basename(args.input.rstrip("/")),
                    "status": "crashed", "reason": f"{type(e).__name__}: {e}",
                }) + "\n")
                continue
            rows, labels, seats = meta.pop("_arrays")
            np.savez_compressed(npz_path, rows=rows, labels=labels,
                                seats=seats)
            mf.write(json.dumps(meta) + "\n")

            status = meta["status"]
            agg[f"status_{status}"] += 1
            agg["episodes"] += 1
            agg["questions"] += meta["n_questions"]
            agg["forced"] += meta["n_forced"]
            agg["decisions"] += meta["n_decisions"]
            agg["rows"] += meta["n_rows"]
            agg["stop_rows"] += meta["n_stop_rows"]
            agg["voluntary_stops"] += meta["voluntary_stops"]
            agg["mismatch_answers"] += meta["n_mismatch_answers"]
            agg["rep_violations"] += meta["rep_violations"]
            agg["mask_violations"] += meta["mask_violations"]
            agg["nonfinite_rows"] += meta["nonfinite_rows"]
            agg["overflow_options"] += meta["overflow_options"]
            agg["option_cap_events"] += meta["option_cap_events"]
            agg["mulligans"] += sum(meta["mulligans"])
            if meta["reward_consistent"] is False:
                agg["reward_inconsistent"] += 1
            elif meta["reward_consistent"] is True:
                agg["reward_consistent_ep"] += 1
            if meta["prize_frames_found"] is False:
                agg["prize_frames_missing"] += 1

    dt = time.time() - t0
    print(f"processed {agg['episodes']} episodes from {args.input} "
          f"in {dt:.1f}s ({dt / max(1, agg['episodes']):.2f} s/ep), "
          f"skipped {agg['skipped_existing']} existing")

    manifest = {
        "schema": {
            "npz": {"rows": f"float32 (n, {OBS_SIZE}) — encoder rows, "
                            "identical layout to ptcg/rl/buffers.py VERSION 5",
                    "labels": "int16 (n,) — class-rep option index or STOP="
                              f"{STOP}",
                    "seats": "int8 (n,) — acting seat 0/1"},
            "episodes.jsonl": "per-episode metadata; quality-gate input",
        },
        "input": args.input, "out": args.out, "limit": args.limit,
        "stats": dict(agg),
    }
    with open(os.path.join(args.out, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    if args.self_check:
        print("== self-check ==")
        for k in sorted(agg):
            print(f"  {k}: {agg[k]}")
        hard = {
            "rep_violations": 0, "mask_violations": 0,
            "nonfinite_rows": 0, "overflow_options": 0,
            "episodes_crashed": 0,
        }
        bad = [f"{k}={agg[k]}" for k in hard if agg[k] != hard[k]]
        if bad:
            print("SELF-CHECK FAIL:", ", ".join(bad), file=sys.stderr)
            return 1
        if agg["episodes"] == 0:
            print("SELF-CHECK FAIL: no episodes extracted", file=sys.stderr)
            return 1
        print("SELF-CHECK OK "
              f"(rows={agg['rows']} decisions={agg['decisions']} "
              f"forced={agg['forced']} stop_rows={agg['stop_rows']} "
              f"mulligans={agg['mulligans']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
