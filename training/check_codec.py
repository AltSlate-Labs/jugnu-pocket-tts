"""Codec reconstruction gate (decision D14): run before any training launch.

Encodes and decodes held-out clips with the codec a training config would use, and
fails unless the latents are non-degenerate and the reconstruction stays intelligible.

    python ~/hindi-tts/check_codec.py <train_config.yaml> <manifest.jsonl> [n_clips]   # from the pocket-tts checkout
"""

import json
import sys

import jiwer
import sphn
import torch
from training.args import load_args
from training.eval.librispeech import latents_to_wav
from training.modules.builders import build_models
from transformers import pipeline

config, manifest = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
device = torch.device("cuda")
args = load_args(config)
args.distill_cfg_coef = 0.0
_, mimi, _ = build_models(args)
mimi = mimi.to(device)
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
               device=device, torch_dtype=torch.float16)  # fmt: skip


def transcribe(wav, sr, lang):
    # Language is forced: on auto-detect whisper flips Hindi clips to Urdu script or translates them.
    wav16 = sphn.resample(wav, src_sample_rate=sr, dst_sample_rate=16000)[: 29 * 16000]
    out = asr({"array": wav16, "sampling_rate": 16000},
              generate_kwargs={"num_beams": 1, "language": lang, "task": "transcribe"})  # fmt: skip
    return out["text"].lower().strip() or "<empty>"


rows = [json.loads(line) for line in open(manifest)][:: max(1, sum(1 for _ in open(manifest)) // n)][:n]
stds, origs, recs = [], [], []
for r in rows:
    start = float(r.get("start", 0.0))
    wav, sr = sphn.read(r["path"], start_sec=start or None, duration_sec=r["duration"] if start else None)
    wav = sphn.resample(wav.mean(0), src_sample_rate=sr, dst_sample_rate=mimi.sample_rate)
    with torch.no_grad():
        lat = mimi.encode_to_latent(torch.from_numpy(wav).float()[None, None].to(device))[0]
    stds.append(float(lat.std()))
    rec = latents_to_wav(mimi, lat, device).cpu().numpy()
    origs.append(transcribe(wav, mimi.sample_rate, r.get("lang", "en")))
    recs.append(transcribe(rec, mimi.sample_rate, r.get("lang", "en")))
# English is the codec's home turf (tight bound); Hindi is 16 kHz field audio judged by a weak
# ASR, so its bound only catches a broken codec, and the number itself is logged as a finding.
LIMITS = {"en": 0.10, "hi": 0.35}
ok = min(stds) > 1e-3
print(f"latent std min {min(stds):.4f}")
for lang, limit in LIMITS.items():
    idx = [i for i, r in enumerate(rows) if r.get("lang", "en") == lang]
    if not idx:
        continue
    drift = jiwer.wer([origs[i] for i in idx], [recs[i] for i in idx])
    print(f"{lang}: ASR drift original->reconstruction WER {drift:.3f} over {len(idx)} clips (limit {limit})")
    ok = ok and drift < limit
print("CODEC GATE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
