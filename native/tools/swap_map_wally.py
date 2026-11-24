"""Regenerate native/ptcg/family_map.h with WALLY's family as an expert.

Default map: 1 abra, 2 dragapult, 3 excadrill, 4 shuppet, 5 ogerpon.
Swapped map: excadrill (floor family, ~4% win rate) -> shared; slot 3 becomes
lopunny_froslass_tempo, the family of the C2 learner deck (WALLY), so the
improved-C2 learner has a dedicated expert head.

Usage: python3 -I native/tools/swap_map_wally.py [--revert]
"""
import csv
import os
import sys

ROOT = os.getcwd()
MAP = "/root/autodl-tmp/tmp/pool_v21_family_map.csv"
OUT = os.path.join(ROOT, "native/ptcg/family_map.h")


def main(revert=False):
    if revert:
        experts = {"abra_kadabra_mill": 1, "dragapult_spread": 2,
                   "excadrill_steel_mill": 3, "shuppet_dhelmise_trap": 4,
                   "ogerpon_grass_accel": 5}
        note = "default 5-expert map"
    else:
        experts = {"abra_kadabra_mill": 1, "dragapult_spread": 2,
                   "lopunny_froslass_tempo": 3, "shuppet_dhelmise_trap": 4,
                   "ogerpon_grass_accel": 5}
        note = "WALLY family (lopunny) in slot 3; excadrill -> shared"
    m = {}
    for r in csv.DictReader(open(MAP)):
        m[r['pool_file']] = experts.get(r['family'], 0)
    files = sorted(f for f in os.listdir("decks/pool") if f.endswith(".csv"))
    assert len(files) == 1369, len(files)
    arr = [m.get(f, 0) for f in files]
    with open(OUT, "w") as fh:
        fh.write(f"// Auto-generated: {note}\n")
        fh.write("// 0 = shared, 1=abra, 2=dragapult, 3=slot3, 4=shuppet, 5=ogerpon.\n")
        fh.write("#ifndef PTCG_FAMILY_MAP_H\n#define PTCG_FAMILY_MAP_H\n#include <stdint.h>\n")
        fh.write("#define PT_N_DECK_FAMILY 1369\nstatic const uint8_t kDeckFamily[1369] = {\n")
        for i in range(0, 1369, 32):
            fh.write("    " + ",".join(str(x) for x in arr[i:i + 32]) + ",\n")
        fh.write("};\n#endif\n")
    print(note, {i: arr.count(i) for i in range(6)})


if __name__ == "__main__":
    main("--revert" in sys.argv)
