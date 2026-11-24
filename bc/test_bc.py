"""bc pipeline end-to-end dry-run test (5 episodes).

    唯一的 torch 触点是「NLL 形状验证」：对 5 个 episode 的行做一次
    batched forward_policy + masked NLL + 一步 backward。这是 1 个 batch
    级别的形状/有限性检查——不构造优化器、不循环、权重即弃——按任务书
    明确不算训练。

Checks per episode (fresh init weights are NEVER involved in extraction —
extraction is pure encoder/tracker):
  1. extraction status == extract_ok, both decks present
  2. 行数对账: questions == decisions + forced; labels == rows;
     steps >= questions + deck_calls
  3. rows finite (np.isfinite all), shape (n, OBS_SIZE)
  4. 动作合法性: every label's opt_mask slot == 1 (STOP=63 needs
     stop_ok, guaranteed by construction and checked in extraction)
  5. class-mirror crosscheck clean (rep_violations == 0)
  6. quality_gate.gate() runs (rating gate skipped w/o --leaderboard,
     deck-share disabled) and passes at least the sane episodes
  7. NLL on a sample batch: finite loss, finite logits, one backward
     step with finite grads  [shape validation — NOT training]

Usage:
  PYTHONPATH=data:. python3 bc/test_bc.py \
      --input /root/autodl-fs/pokemon_RL_old_bc/bc_extract/2026-11-05 \
      [--limit 5] [--out /root/autodl-tmp/tmp/bc_test_out]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "data"))
sys.path.insert(0, os.path.join(REPO, "bc"))

import numpy as np

import extract as extract_mod
import quality_gate


def main():
    ap = argparse.ArgumentParser(description="bc pipeline dry-run test")
    ap.add_argument("--input", default="/root/autodl-fs/pokemon_RL_old_bc"
                                       "/bc_extract/2026-11-05")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--out", default="/root/autodl-tmp/tmp/bc_test_out")
    ap.add_argument("--device", default=None)
    ap.add_argument("--keep-out", action="store_true",
                    help="do not delete the temp output dir on success")
    args = ap.parse_args()

    from ptcg.rl.buffers import OBS_SIZE, OFFSETS, MAX_OPTIONS, STOP

    files = sorted(f for f in os.listdir(args.input)
                   if f.endswith(".json"))[:args.limit]
    assert len(files) == args.limit, \
        f"need {args.limit} episodes, found {len(files)} in {args.input}"
    os.makedirs(args.out, exist_ok=True)
    a0, a1 = OFFSETS["opt_mask"]

    metas = []
    totals = dict(rows=0, decisions=0, forced=0, questions=0,
                  rep_bad=0, mask_bad=0, nonfinite=0, mism=0, stops=0)
    per_ep = []
    for name in files:
        episode_id = name[:-5]
        meta = extract_mod.extract_episode(os.path.join(args.input, name),
                                           episode_id,
                                           os.path.basename(args.input))
        assert meta["status"] == "extract_ok", \
            f"{episode_id}: status={meta['status']} reason={meta['reason']}"
        rows, labels, seats = meta.pop("_arrays")
        # 2) 行数对账
        assert meta["n_questions"] == meta["n_decisions"] + meta["n_forced"], \
            f"{episode_id}: question/decision accounting"
        assert len(labels) == len(rows) == meta["n_rows"], \
            f"{episode_id}: labels/rows count"
        assert meta["n_steps"] >= meta["n_deck_calls"], \
            f"{episode_id}: steps vs deck calls"
        # 3) 有限性 + 形状
        assert rows.dtype == np.float32 and rows.shape[1] == OBS_SIZE, \
            f"{episode_id}: row shape {rows.shape}"
        assert np.isfinite(rows).all(), f"{episode_id}: non-finite rows"
        assert set(np.unique(seats)).issubset({0, 1}), \
            f"{episode_id}: seats"
        # 4) 动作合法性: label slot in opt_mask; labels in [0, STOP]
        assert labels.min() >= 0 and labels.max() <= STOP, \
            f"{episode_id}: label range"
        masks = rows[:, a0:a1]
        lab_ok = masks[np.arange(len(labels)), labels.astype(np.int64)] > 0
        assert lab_ok.all(), \
            f"{episode_id}: {int((~lab_ok).sum())} labels not mask-legal"
        # 5) mirror crosscheck clean
        assert meta["rep_violations"] == 0, f"{episode_id}: rep_violations"
        assert meta["mask_violations"] == 0, f"{episode_id}: mask_violations"
        assert meta["n_mismatch_answers"] == 0, \
            f"{episode_id}: mismatched recorded answers"
        assert meta["reward_consistent"] is True, \
            f"{episode_id}: rewards vs frames result"
        for k in totals:
            key = {"rows": "n_rows", "decisions": "n_decisions",
                   "forced": "n_forced", "questions": "n_questions",
                   "rep_bad": "rep_violations",
                   "mask_bad": "mask_violations",
                   "nonfinite": "nonfinite_rows",
                   "mism": "n_mismatch_answers",
                   "stops": "n_stop_rows"}[k]
            totals[k] += meta.get(key, 0)
        per_ep.append((episode_id, meta["n_steps"], meta["n_questions"],
                       meta["n_forced"], meta["n_decisions"], len(rows),
                       meta["mulligans"], meta["turns"]))
        metas.append(meta)
        np.savez_compressed(os.path.join(args.out, f"{episode_id}.npz"),
                            rows=rows, labels=labels, seats=seats)
        with open(os.path.join(args.out, "episodes.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")

    print("== per-episode reconciliation (steps, questions, forced, "
          "decisions, rows, mulligans, turns) ==")
    for t in per_ep:
        print("  ", t)

    # 6) quality gate on the extracted metas (no leaderboard -> rating gate
    #    skipped; deck-share disabled: a 5-episode sample is far below the
    #    share cap by construction)
    class A:                                  # minimal args namespace
        leaderboard = None
        min_rating = 1650.0
        rating_rule = "both"
        max_deck_share = 1.0
        max_mulligans = 3
        min_turns = 4
        max_turns = 30
        min_rows = 10
        allow_reward_mismatch = False
    records, agg = quality_gate.gate(metas, A)
    n_pass = sum(r["passed"] for r in records)
    assert n_pass >= max(1, len(records) - 1), \
        f"quality gate passed only {n_pass}/{len(records)}: {agg}"
    print(f"== quality gate: {n_pass}/{len(records)} passed "
          f"(rating gate skipped: no leaderboard; deck-share disabled) ==")

    # 7) NLL shape validation — ONE forward + ONE backward on one batch.
    #    不是训练：无优化器、无循环、权重即弃。
    import torch
    from ptcg.rl.model import PTCGTransformer
    torch.manual_seed(42)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = PTCGTransformer(critic="oracle").to(device).train(False)
    rows_all = np.concatenate([
        np.load(os.path.join(args.out, f"{m['episode_id']}.npz"))["rows"]
        for m in metas])
    labels_all = np.concatenate([
        np.load(os.path.join(args.out, f"{m['episode_id']}.npz"))["labels"]
        for m in metas])
    rng = np.random.default_rng(7)
    pick = rng.permutation(len(labels_all))[:64]
    R = torch.from_numpy(rows_all[pick]).to(device)
    L = torch.from_numpy(labels_all[pick].astype(np.int64)).to(device)
    M = torch.from_numpy(rows_all[pick][:, a0:a1]).to(device)
    with torch.no_grad():
        logits = model.forward_policy(R)
    assert torch.isfinite(logits).all(), "non-finite logits"
    masked = logits.masked_fill(~M.bool(), float("-inf"))
    loss_full = torch.nn.functional.cross_entropy(masked.float(), L,
                                                  reduction="sum")
    assert torch.isfinite(loss_full).item(), "non-finite NLL"
    R.requires_grad_(False)
    logits2 = model.forward_policy(R)
    masked2 = logits2.masked_fill(~M.bool(), float("-inf"))
    loss = torch.nn.functional.cross_entropy(masked2.float(), L)
    loss.backward()
    gsum = 0.0
    gnon = 0
    gfinite = True
    for p in model.parameters():
        if p.grad is not None:
            gsum += float(p.grad.abs().sum())
            gnon += int(p.grad.abs().sum() > 0)
            gfinite = gfinite and bool(torch.isfinite(p.grad).all())
    assert gnon > 0 and gfinite, "no finite gradients after one backward"
    acc = (logits2.argmax(-1) == L).float().mean().item()
    print(f"== NLL shape check (batch=64, {device}, fresh seed-42 init; "
          "NOT training — no optimizer, weights discarded) ==")
    print(f"  batch NLL={loss.item():.4f} "
          f"(sum {loss_full.item():.1f}) acc={acc:.3f}")
    print(f"  grads: {gnon} params with nonzero grad, "
          f"|g|_1={gsum:.4e}, all finite: {gfinite}")

    if not args.keep_out:
        shutil.rmtree(args.out, ignore_errors=True)
    print(f"DRY-RUN OK: {len(files)} episodes, {totals['rows']} rows, "
          f"{totals['decisions']} decisions, {totals['forced']} forced, "
          f"{totals['stops']} stop rows, 0 violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
