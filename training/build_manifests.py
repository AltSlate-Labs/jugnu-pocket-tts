"""Manifest steps around forced alignment.

  split    <iv_dir>                      all.jsonl -> hi_{train,valid,test}.jsonl (speaker-disjoint)
  finalize <iv_dir> <en_dir> <out_dir>   swap Hinglish words into aligned Hindi, merge with English
"""

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

N_HOLDOUT = 20  # speakers each for valid and test, half per gender
MAX_VALID_UTTS = 400  # per language; validation only drives loss curves and samples


def read(p):
    return [json.loads(line) for line in open(p)]


def write(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hours = sum(r["duration"] for r in rows) / 3600
    print(f"{p}: {len(rows)} utt, {hours:.1f} h, {len({r.get('speaker') for r in rows})} spk")


def split(iv_dir: Path):
    rows = read(iv_dir / "all.jsonl")
    root = iv_dir.resolve().parent.parent  # manifest paths were written relative to the run dir
    by_spk = defaultdict(list)
    for r in rows:
        if not Path(r["path"]).is_absolute():
            r["path"] = str(root / r["path"])
        by_spk[r["speaker"]].append(r)
    rng = random.Random(0)
    held = {"valid": [], "test": []}
    for gender in ("Male", "Female"):
        # Holdout speakers need enough audio to serve as both reference and target.
        cands = sorted(
            s for s, rs in by_spk.items()
            if rs[0]["gender"] == gender and 10 <= len(rs) <= 150
        )  # fmt: skip
        rng.shuffle(cands)
        held["valid"] += cands[: N_HOLDOUT // 2]
        held["test"] += cands[N_HOLDOUT // 2 : N_HOLDOUT]
    out = {"train": [], "valid": [], "test": []}
    for s, rs in by_spk.items():
        part = "valid" if s in held["valid"] else "test" if s in held["test"] else "train"
        out[part] += rs
    for part, rs in out.items():
        write(iv_dir / f"hi_{part}.jsonl", rs)
    json.dump(held, open(iv_dir / "holdout_speakers.json", "w"), indent=1)


def swap_mixed(r: dict) -> dict:
    """Use the Latin spelling of glossed English words for a deterministic half of utterances."""
    mw = r.pop("mixed_words", None)
    if not mw or len(mw) != len(r["words"]):
        return r
    if int(hashlib.md5(r["path"].encode()).hexdigest(), 16) % 2:
        return r
    for w, m in zip(r["words"], mw, strict=True):
        w["word"] = m
    r["transcript"] = " ".join(mw)
    r["mixed"] = True
    return r


def finalize(iv_dir: Path, en_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    for part in ("train", "valid"):
        hi = [swap_mixed(r) for r in read(iv_dir / f"hi_{part}_aligned.jsonl")]
        en = read(en_dir / f"{part}_aligned.jsonl")
        for r in hi:
            r["lang"] = "hi"
        for r in en:
            r["lang"] = "en"
        if part == "valid":
            rng.shuffle(hi), rng.shuffle(en)
            hi, en = hi[:MAX_VALID_UTTS], en[:MAX_VALID_UTTS]
        print(f"{part}: mixed-script utterances = {sum(bool(r.get('mixed')) for r in hi)}")
        rows = hi + en
        rng.shuffle(rows)
        write(out_dir / f"{part}_aligned.jsonl", rows)


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"split": split, "finalize": finalize}[cmd](*map(Path, args))
