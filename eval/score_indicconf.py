"""Re-score an eval_tts.py output dir with IndicConformer-600M (RNNT, CPU, nemoenv).

Whisper is a weak judge of this speech (its WER on the real recordings is ~40%),
so IndicConformer is the primary Hindi/Hinglish intelligibility metric.

    ~/hindi-stt/nemoenv/bin/python score_indicconf.py <eval_out_dir>
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

import jiwer
import librosa
import soundfile as sf
import torch
from transformers import AutoModel

REAL_CACHE = Path("/home/ubuntu/hindi-tts/data/iv_hi/asr_real_indicconf.json")


def normalize(text: str) -> str:  # same as eval_tts.normalize
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("़", "").replace("ँ", "ं")
    text = re.sub(r"[^\w\sऀ-ॿ]|[।॥_]", " ", text)
    return " ".join(text.split())


def main(out_dir: Path):
    items = [json.loads(line) for line in open(out_dir / "items.jsonl")]
    model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

    def transcribe(path: str) -> str:
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        try:
            return normalize(model(torch.tensor(wav[: 30 * 16000])[None], "hi", "rnnt"))
        except Exception:  # noqa: BLE001 -- the ONNX wrapper raises on degenerate audio
            return ""

    real = json.load(open(REAL_CACHE)) if REAL_CACHE.exists() else {}
    for c in items:
        c["ic_hyp"] = transcribe(c["wav"]) if c.get("wav") else ""
        if c["ref"] not in real:
            real[c["ref"]] = transcribe(c["ref"])
        c["ic_hyp_real"] = real[c["ref"]]
    json.dump(real, open(REAL_CACHE, "w"), ensure_ascii=False)

    results = {}
    for name in ("hindi", "hinglish"):
        rows = [c for c in items if c["set"] == name]
        refs = [c["ref_norm"] for c in rows]
        results[name] = {
            "n": len(rows),
            "wer": round(jiwer.wer(refs, [c["ic_hyp"] for c in rows]), 4),
            "cer": round(jiwer.cer(refs, [c["ic_hyp"] for c in rows]), 4),
            "wer_real_audio": round(jiwer.wer(refs, [c["ic_hyp_real"] for c in rows]), 4),
            "cer_real_audio": round(jiwer.cer(refs, [c["ic_hyp_real"] for c in rows]), 4),
        }
    json.dump(results, open(out_dir / "results_indicconf.json", "w"), indent=1)
    with open(out_dir / "items.jsonl", "w") as f:
        for c in items:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
