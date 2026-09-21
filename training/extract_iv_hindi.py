"""Extract a TTS-clean Hindi subset of IndicVoices into flac + a pocket-tts manifest.

Filters on the dataset's own verification flags, drops utterances with non-speech
tags, and records a mixed-script (Hinglish) word list built from the [English]
glosses in `unsanitized_normalized`. Alignment runs on the Devanagari transcript;
`mixed_words` is swapped in afterwards (see finalize_manifests.py).

Usage: python extract_iv_hindi.py <parquet_dir> <out_dir>
"""

import ast
import glob
import io
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

BAD_FLAGS = [
    "low_volume", "noise_intermittent", "noise_persistent", "chatter_intermittent",
    "chatter_persistent", "unclear_audio", "off_topic", "repeating_content", "long_pauses",
    "mispronunciation", "stretching", "bad_extempore_quality", "skipping_words",
    "incorrect_text_prompt", "wrong_language", "echo_present", "sst",
]  # fmt: skip
MIN_SEC, MAX_SEC = 2.0, 30.0
COLS = ["audio_filepath", "duration", "normalized", "unsanitized_normalized", "speaker_id",
        "gender", "age_group", "scenario", "verification_report"]  # fmt: skip


def mixed_words(unsanitized: str, words: list[str]) -> list[str] | None:
    """Words with single-word [English] glosses swapped in; None if nothing to swap
    or the gloss stream does not line up with the clean transcript."""
    toks = re.findall(r"\[[^\]]*\]|<[^>]*>|\S+", unsanitized)
    out, swapped = [], False
    for t in toks:
        if t.startswith("<"):
            continue
        if t.startswith("["):
            gloss = t[1:-1].strip()
            if out and re.fullmatch(r"[A-Za-z][A-Za-z'.-]*", gloss):
                out[-1] = ("en", out[-1][1], gloss)
                swapped = True
            continue
        out.append(("hi", t, t))
    if not swapped or [o[1] for o in out] != words:
        return None
    return [o[2] for o in out]


def do_shard(args: tuple[str, str]) -> dict:
    shard, out_dir = args
    name = Path(shard).stem
    audio_dir = Path(out_dir) / "audio" / name
    audio_dir.mkdir(parents=True, exist_ok=True)
    stats = {"seen": 0, "kept": 0, "hours": 0.0, "mixed": 0}
    pf = pq.ParquetFile(shard)
    with open(Path(out_dir) / "parts" / f"{name}.jsonl", "w") as fout:
        i = -1
        for batch in pf.iter_batches(batch_size=256, columns=COLS):
            for r in batch.to_pylist():
                i += 1
                stats["seen"] += 1
                text = (r["normalized"] or "").strip()
                if not (MIN_SEC <= r["duration"] <= MAX_SEC) or not text or "<" in text:
                    continue
                try:
                    rep = ast.literal_eval(r["verification_report"])
                except (ValueError, SyntaxError):
                    continue
                if rep.get("decision") != "excellent" or any(rep.get(k) for k in BAD_FLAGS):
                    continue
                words = text.split()
                if len(words) < 3:
                    continue
                wav, sr = sf.read(io.BytesIO(r["audio_filepath"]["bytes"]), dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                path = audio_dir / f"{i:06d}.flac"
                sf.write(path, wav, sr)
                rec = {
                    "path": str(path),
                    "duration": round(len(wav) / sr, 3),
                    "transcript": text,
                    "speaker": r["speaker_id"],
                    "gender": r["gender"],
                    "age_group": r["age_group"],
                    "scenario": r["scenario"],
                }
                mw = mixed_words(r["unsanitized_normalized"] or "", words)
                if mw:
                    rec["mixed_words"] = mw
                    stats["mixed"] += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["kept"] += 1
                stats["hours"] += rec["duration"] / 3600
    return stats


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    (Path(out) / "parts").mkdir(parents=True, exist_ok=True)
    shards = sorted(glob.glob(f"{src}/*.parquet"))
    total = {"seen": 0, "kept": 0, "hours": 0.0, "mixed": 0}
    with Pool(40) as pool:
        for s in pool.imap_unordered(do_shard, [(s, out) for s in shards]):
            for k in total:
                total[k] += s[k]
    with open(Path(out) / "all.jsonl", "w") as f:
        for p in sorted(glob.glob(f"{out}/parts/*.jsonl")):
            f.write(open(p).read())
    print(json.dumps(total))
