"""Compares two EXP_MESH_DUMP files (patches/exp-mesh-dump.patch) as multisets of emitted-vertex records.

Usage: python3 -I tools/mesh_dump_diff.py <dump A> <dump B> [--json=<out>]
Prints the records each file has that the other lacks; --json writes the counts and the first differing records.
"""
import json
import struct
import sys
from collections import Counter


def records(path):
    """Returns a dump's recorded count, its record size in words, and its records as a Counter of word tuples."""
    data = open(path, "rb").read()
    count, size = struct.unpack_from("<II", data)
    body = struct.unpack_from(f"<{len(data) // 4}I", data)[4:]
    return count, size, Counter(tuple(body[i:i + size]) for i in range(0, len(body), size))


def show(record):
    """One record as text: its four header words, then its output registers as floats."""
    regs = [struct.unpack("<f", struct.pack("<I", w))[0] for w in record[4:]]
    return (f"workload {record[0]} prim {record[1]} vertex {record[2]} before {record[3]} | "
            + " ".join(f"{v:.4g}" for v in regs))


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = next((a[7:] for a in sys.argv[1:] if a.startswith("--json=")), None)
    (ca, sa, ra), (cb, sb, rb) = records(paths[0]), records(paths[1])
    only_a, only_b = ra - rb, rb - ra
    print(f"A {ca} records of {sa} words, B {cb} records of {sb} words")
    print(f"only in A: {sum(only_a.values())}, only in B: {sum(only_b.values())}")
    for name, only in (("A", only_a), ("B", only_b)):
        for record in sorted(only)[:6]:
            print(f"  {name}: {show(record)}")
    if out:
        with open(out, "w") as f:
            json.dump({"a": paths[0], "b": paths[1], "records": [ca, cb], "words": [sa, sb],
                       "only_a": sum(only_a.values()), "only_b": sum(only_b.values()),
                       "first_only_a": [show(r) for r in sorted(only_a)[:6]],
                       "first_only_b": [show(r) for r in sorted(only_b)[:6]]}, f, indent=1)


if __name__ == "__main__":
    main()
