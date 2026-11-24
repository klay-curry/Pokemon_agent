"""BC training (masked NLL) on top2 decision rows — DELIVERED, NOT RUN.

    本脚本按任务书只交付、不运行。运行它 = 训练；本次任务的执行边界是
    `python3 bc/train_bc.py --help`。bc/test_bc.py 里的单次 forward + 一步
    反传是形状验证（1-2 条样本级别），明确不属于训练。

What it does when someone later chooses to run it:
  * loads bc/extract.py output (<data>/<epid>.npz rows/labels/seats +
    episodes.jsonl; optional --keep passed.jsonl from bc/quality_gate.py);
  * loads the top2 policy: ptcg/rl/model.PTCGTransformer(critic="oracle"),
    optionally warm-started from a native fp32 blob via
    native/tools/native_weights.py blob_to_sd (e.g. parity_init.bin) —
    this is the same blob format train_native.py/PPO consumes;
  * trains masked NLL over the encoder's equivalence-class-deduped
    option categories: the labels ARE class-representative indices
    (extract.py answer_label), and the logits are masked to legal classes
    (opt_mask) before the softmax — i.e. CE over classes, never raw option
    indices;
  * streams episodes in shuffled chunks (rows never fully resident; a full
    4-day extract is >100 GB of rows), batch 64, AdamW, grad-clip 1.0,
    --epochs 2 default, episode-hash val split, terminal logs only;
  * exports at the end:
      OUT/ckpt.pt          torch state_dict + args (full model)
      OUT/bc_init.bin      top2 packed fp32 blob via
                           native_weights.sd_to_blob — drop-in PPO init
                           (native_weights.py export CKPT.pt equivalent);
      OUT/train_summary.json

Usage (when authorized):
  PYTHONPATH=data:. python3 bc/train_bc.py \
      --data /root/autodl-tmp/bc_out/2026-11-05 \
      --keep  /root/autodl-tmp/bc_out/gate_2026-11-05/passed.jsonl \
      --out  /root/autodl-tmp/bc_ckpt/bc_s1 \
      --init /root/autodl-tmp/top2_repro/parity_init.bin --epochs 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "data"))
sys.path.insert(0, os.path.join(REPO, "native", "tools"))

import numpy as np


# ---------------------------------------------------------------------------
# data: streaming episode-chunk loader over bc/extract.py output
# ---------------------------------------------------------------------------
def load_manifest(data_dir: str, keep_ids: set[str] | None):
    metas = []
    with open(os.path.join(data_dir, "episodes.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("status") != "extract_ok":
                continue
            if keep_ids is not None and m.get("episode_id") not in keep_ids:
                continue
            metas.append(m)
    return metas


def read_keep(path: str | None) -> set[str] | None:
    if not path:
        return None
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["episode_id"])
    return ids


def chunk_batches(data_dir: str, metas, batch: int, seed: int, epoch: int,
                  val_ids: set[str], train: bool, chunk_eps: int = 256):
    """Yield (rows f32 (B,OBS_SIZE), labels i64 (B), masks f32 (B,64)) by
    streaming shuffled episode chunks; rows of one chunk live in RAM only."""
    rng = np.random.default_rng(seed * 100003 + epoch * 131 + (0 if train
                                                                else 7))
    import torch                       # local: module stays --help-light
    order = list(range(len(metas)))
    rng.shuffle(order)
    pool_r, pool_l, pool_m = [], [], []
    a0, a1 = None, None
    from ptcg.rl.buffers import OFFSETS, MAX_OPTIONS
    a0, a1 = OFFSETS["opt_mask"]

    def flush():
        nonlocal pool_r, pool_l, pool_m
        if not pool_r:
            return
        R = np.concatenate(pool_r); L = np.concatenate(pool_l)
        M = np.concatenate(pool_m)
        pool_r, pool_l, pool_m = [], [], []
        idx = rng.permutation(len(L))
        for i in range(0, len(idx), batch):
            j = idx[i:i + batch]
            if len(j) < batch:            # drop ragged tail
                continue
            yield (torch.from_numpy(R[j]),
                   torch.from_numpy(L[j].astype(np.int64)),
                   torch.from_numpy(M[j]))

    for pos, mi in enumerate(order):
        m = metas[mi]
        is_val = m["episode_id"] in val_ids
        if is_val != (not train):
            continue
        z = np.load(os.path.join(data_dir, f"{m['episode_id']}.npz"))
        rows = z["rows"]
        labels = z["labels"].astype(np.int64)
        masks = rows[:, a0:a1]
        pool_r.append(rows); pool_l.append(labels); pool_m.append(masks)
        if len(pool_r) >= chunk_eps:
            yield from flush()
    yield from flush()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="BC masked-NLL training on top2 rows (delivered, "
                    "not run as part of this task)")
    ap.add_argument("--data", required=True,
                    help="bc/extract.py output dir")
    ap.add_argument("--keep", default=None,
                    help="quality_gate.py passed.jsonl (optional)")
    ap.add_argument("--out", required=True, help="ckpt + blob output dir")
    ap.add_argument("--init", default=None,
                    help="native fp32 blob to warm-start (e.g. "
                         "parity_init.bin); omit for fresh seed-42 init")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.02,
                    help="episode-level val split (hash-based)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None,
                    help="cuda|cpu (default: cuda if available)")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from ptcg.rl.buffers import OBS_SIZE, OFFSETS, MAX_OPTIONS
    from ptcg.rl.model import PTCGTransformer
    from native_weights import blob_to_sd, sd_to_blob, D, HEADS, FFN, LAYERS

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    keep_ids = read_keep(args.keep)
    metas = load_manifest(args.data, keep_ids)
    if not metas:
        print("no usable episodes under", args.data, file=sys.stderr)
        return 2
    total_rows = sum(m["n_rows"] for m in metas)
    val_ids = {m["episode_id"] for m in metas
               if int(hashlib.md5(m["episode_id"].encode())
                      .hexdigest(), 16) % 10000 < args.val_frac * 10000}
    print(f"episodes={len(metas)} rows={total_rows} val_eps={len(val_ids)} "
          f"device={device}")

    # The native trunk is d128/h8/ffn512/L4 (native_weights D/HEADS/FFN/
    # LAYERS). PTCGTransformer defaults to d=256, so building it without
    # these args trains a d256 model whose sd_to_blob export fails on
    # token_proj (256,324) != (128,324) -- the delivered script's original
    # bug, hit on 2026-11-22 when it was first run end-to-end.
    model = PTCGTransformer(d=D, layers=LAYERS, heads=HEADS, ffn=FFN,
                            critic="oracle").to(device)
    if args.init:
        blob = np.fromfile(args.init, dtype=np.float32)
        sd = model.state_dict()
        sd.update(blob_to_sd(blob))
        model.load_state_dict(sd)
        print(f"warm start from {args.init} ({blob.size} params)")
    a0, a1 = OFFSETS["opt_mask"]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)

    def masked_nll(logits, labels, mask):
        # categories are already equivalence-class reps (extract.py); mask
        # non-legal classes before the softmax (model itself pre-fills
        # -1e9, this makes it explicit and idempotent)
        logits = logits.masked_fill(~mask.bool(), float("-inf"))
        return F.cross_entropy(logits, labels)

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        for rows, labels, masks in chunk_batches(
                args.data, metas, args.batch, args.seed, epoch, val_ids,
                train=True):
            rows = rows.to(device); labels = labels.to(device)
            masks = masks.to(device)
            logits = model.forward_policy(rows)
            loss = masked_nll(logits, labels, masks)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                with torch.no_grad():
                    acc = (logits.argmax(-1) == labels).float().mean()
                print(f"ep{epoch} step{step} nll={loss.item():.4f} "
                      f"acc={acc.item():.3f} ({time.time() - t0:.0f}s)")
        # val pass
        model.eval()
        tot, n, correct = 0.0, 0, 0
        with torch.no_grad():
            for rows, labels, masks in chunk_batches(
                    args.data, metas, args.batch, args.seed, epoch, val_ids,
                    train=False):
                logits = model.forward_policy(rows.to(device))
                m = masks.to(device)
                tot += masked_nll(logits, labels.to(device), m).item() \
                    * len(labels)
                n += len(labels)
                correct += (logits.argmax(-1).cpu() == labels).sum().item()
        print(f"ep{epoch} val nll={tot / max(1, n):.4f} "
              f"acc={correct / max(1, n):.3f} (n={n})")

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                "obs_size": OBS_SIZE}, ckpt_path)
    blob = sd_to_blob(model.state_dict())
    blob_path = os.path.join(args.out, "bc_init.bin")
    blob.astype(np.float32).tofile(blob_path)
    with open(os.path.join(args.out, "train_summary.json"), "w") as f:
        json.dump({"steps": step, "epochs": args.epochs,
                   "episodes": len(metas), "rows": total_rows,
                   "final_batch_nll": float(loss.item()),
                   "blob_params": int(blob.size),
                   "obs_size": OBS_SIZE}, f, indent=1)
    print(f"saved {ckpt_path} and {blob_path} ({blob.size} params)")


if __name__ == "__main__":
    sys.exit(main())
