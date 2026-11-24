"""Convert a scalar-value native blob (200 params) to the HL-Gauss layout
(202: + hl_value_bins w/b). The bins head starts zero-weight with a Gaussian
bias centred at 0, so the initial expectation is ~0 and the head learns.

Usage: python3 -I native/tools/make_hl_init.py IN.bin OUT.bin [--sigma 0.02]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "data"))

K = 51
VMIN, VMAX = -1.2, 1.2


def main(src, dst, sigma=0.02):
    os.environ.pop("PTCG_HL", None)
    from native.tools.native_weights import blob_to_sd, sd_to_blob
    old = blob_to_sd(np.fromfile(src, dtype=np.float32))
    centers = VMIN + (VMAX - VMIN) * np.arange(K) / (K - 1)
    bias = -(centers ** 2) / (2.0 * sigma * sigma)
    old["hl_value_bins.weight"] = torch.zeros(K, 256)
    old["hl_value_bins.bias"] = torch.from_numpy(bias.astype(np.float32))
    os.environ["PTCG_HL"] = "1"
    out = sd_to_blob(old)
    out.tofile(dst)
    print(f"wrote {dst}: {out.size} floats (HL, K={K})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
