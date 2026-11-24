"""Expand a 120-param C1 blob into the 160-param MoE layout.

Each of the 5 expert heads is initialised as a copy of the shared
score_mlp / oracle_value_mlp, so PTCG_MOE=1 starts behaviourally identical
to C1 at step 0 (routing only diverges as training updates the experts).

Usage: python3 -I native/tools/expand_moe_init.py IN.bin OUT.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "data"))

from native.tools.native_weights import param_order  # noqa: E402


def main(src, dst):
    order = param_order()
    offs, off = {}, 0
    for n, s in order:
        ne = int(np.prod(s))
        offs[n] = (off, ne)
        off += ne
    old = [(n, s) for n, s in order if not n.startswith("moe.")]
    old_total = sum(int(np.prod(s)) for n, s in old)
    blob = np.fromfile(src, dtype=np.float32)
    if blob.size != old_total:
        raise SystemExit(f"{src}: {blob.size} floats, expected {old_total} "
                         f"(is this a 120-param C1 blob?)")
    out = np.zeros(off, dtype=np.float32)
    out[:old_total] = blob
    pairs = [
        ("score.0.weight", "score_mlp.0.weight"),
        ("score.0.bias", "score_mlp.0.bias"),
        ("score.2.weight", "score_mlp.2.weight"),
        ("score.2.bias", "score_mlp.2.bias"),
        ("ov.0.weight", "oracle_value_mlp.0.weight"),
        ("ov.0.bias", "oracle_value_mlp.0.bias"),
        ("ov.2.weight", "oracle_value_mlp.2.weight"),
        ("ov.2.bias", "oracle_value_mlp.2.bias"),
        # full-chain experts: per-family option projections + query MLP
        ("opt_card.weight", "opt_card.weight"),
        ("opt_card.bias", "opt_card.bias"),
        ("opt_attack.weight", "opt_attack.weight"),
        ("opt_attack.bias", "opt_attack.bias"),
        ("q_mlp.0.weight", "q_mlp.0.weight"),
        ("q_mlp.0.bias", "q_mlp.0.bias"),
        ("q_mlp.2.weight", "q_mlp.2.weight"),
        ("q_mlp.2.bias", "q_mlp.2.bias"),
    ]
    for i in range(5):
        for suf, shared in pairs:
            so, sn = offs[shared]
            do, dn = offs[f"moe.{i}.{suf}"]
            assert sn == dn, (suf, sn, dn)
            out[do:do + dn] = blob[so:so + sn]
    out.tofile(dst)
    print(f"wrote {dst}: {out.size} floats "
          f"({old_total} C1 + {out.size - old_total} expert)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
