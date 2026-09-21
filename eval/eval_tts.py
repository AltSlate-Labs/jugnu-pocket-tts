"""Hindi / Hinglish eval for a pocket-tts training run, on held-out speakers.

Builds cross-utterance cloning items from the aligned test manifest (prompt and
target are different utterances of the same unseen speaker), generates, and scores:
  - WER / CER: whisper-large-v3-turbo (language=hi) transcript vs the Devanagari reference.
    Hinglish items are synthesised from mixed-script text but scored against the
    Devanagari transcript, since the ASR writes English words in Devanagari too.
  - the same ASR on the real target recordings, as the ASR-error floor.
  - speaker similarity: wavlm-base-plus-sv cosine, generated vs prompt speaker's target clip.
  - UTMOS if utmos_pytorch is installed.

Run from the pocket-tts checkout:
    python ~/hindi-tts/eval_tts.py runs/scratch --test-jsonl .../hi_test_aligned.jsonl --use-ema
"""

import argparse
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import jiwer
import sphn
import torch
from training.eval.librispeech import latents_to_wav, load_16k, load_mono, load_run

VOICE_SEC = 5.0


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("़", "").replace("ँ", "ं")  # nukta, chandrabindu -> anusvara
    text = re.sub(r"[^\w\sऀ-ॿ]|[।॥_]", " ", text)
    return " ".join(text.split())


def build_items(test_jsonl: str, n_per_set: int, seed: int = 0) -> list[dict]:
    by_spk = defaultdict(list)
    for line in open(test_jsonl):
        r = json.loads(line)
        by_spk[r["speaker"]].append(r)
    rng = random.Random(seed)
    sets = {"hindi": [], "hinglish": []}
    for spk, rows in sorted(by_spk.items()):
        prompts = [r for r in rows if r["duration"] >= VOICE_SEC + 0.5]
        for r in rows:
            if not (3.0 <= r["duration"] <= 12.0):
                continue
            cands = [p for p in prompts if p["path"] != r["path"]]
            if not cands:
                continue
            mixed = r.get("mixed_words")
            sets["hinglish" if mixed else "hindi"].append(
                {
                    "set": "hinglish" if mixed else "hindi",
                    "speaker": spk,
                    "gender": r["gender"],
                    "prompt": rng.choice(cands)["path"],
                    "ref": r["path"],
                    "text": " ".join(mixed) if mixed else r["transcript"],
                    "ref_text": r["transcript"],
                }
            )
    items = []
    for rows in sets.values():
        rng.shuffle(rows)
        items += rows[:n_per_set]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--n-per-set", type=int, default=150)
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--n-steps", type=int, default=1)
    ap.add_argument("--eos-threshold", type=float, default=-1.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--score-device", default="cuda")
    # turbo (1.6 GB fp16) fits beside a training job on a 32 GB card; large-v3 does not.
    ap.add_argument("--asr", default="openai/whisper-large-v3-turbo")
    args = ap.parse_args()

    device = torch.device("cuda")
    items = build_items(args.test_jsonl, args.n_per_set)
    model, mimi, step = load_run(args.run_dir, device, use_ema=args.use_ema, checkpoint=args.checkpoint)
    torch.manual_seed(0)
    out_dir = Path(args.out or f"{args.run_dir}/eval_hi/step{step:08d}_cfg{args.cfg}")
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)

    sp_encode = model.flow_lm.conditioner.tokenizer.sp.encode
    max_frames = int(30.0 * mimi.frame_rate)
    gens: list = [None] * len(items)
    for s in range(0, len(items), args.batch_size):
        chunk = items[s : s + args.batch_size]
        tokens = [torch.tensor(sp_encode(c["text"]), dtype=torch.long) for c in chunk]
        with torch.no_grad():
            voices = [
                mimi.encode_to_latent(
                    load_mono(c["prompt"], mimi.sample_rate)[: int(VOICE_SEC * mimi.sample_rate)][
                        None, None
                    ].to(device)
                )[0]
                for c in chunk
            ]
            outs = model.generate(
                tokens, voices, max_frames=max_frames, temp=args.temp, n_steps=args.n_steps,
                cfg_coef=args.cfg, eos_threshold=args.eos_threshold,
            )  # fmt: skip
        for i, (c, lat) in enumerate(zip(chunk, outs, strict=True)):
            c["no_eos"] = int(lat.shape[0] >= max_frames)
            wav = latents_to_wav(mimi, lat, device)
            if wav is None:
                continue
            path = out_dir / "wav" / f"{s + i:04d}_{c['set']}.wav"
            sphn.write_wav(str(path), wav.cpu().numpy(), int(mimi.sample_rate))
            gens[s + i] = str(path)
    del model
    torch.cuda.empty_cache()

    from transformers import AutoFeatureExtractor, WavLMForXVector, pipeline

    device = torch.device(args.score_device)
    asr = pipeline("automatic-speech-recognition", model=args.asr, device=device,
                   torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)  # fmt: skip

    def transcribe(paths: list[str]) -> list[str]:
        # Cap below whisper's 30 s window: the pipeline cannot batch long-form clips, and a
        # generation that ran to the cap is already counted as a no_eos failure.
        wavs = [{"array": load_16k(p, torch.device("cpu")).numpy()[: 29 * 16000], "sampling_rate": 16000} for p in paths]
        outs = asr(wavs, batch_size=4, generate_kwargs={"language": "hi", "task": "transcribe", "num_beams": 1})
        return [normalize(o["text"]) for o in outs]

    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()

    def embed(path: str) -> torch.Tensor:
        x = fe(load_16k(path, torch.device("cpu")).numpy(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            return sv(**{k: v.to(device) for k, v in x.items()}).embeddings[0]

    try:
        from utmos_pytorch import UTMOSScoreTorch

        utmos = UTMOSScoreTorch(device=str(device))
    except ImportError:
        utmos = None

    ok = [i for i, g in enumerate(gens) if g]
    hyps = dict(zip(ok, transcribe([gens[i] for i in ok]), strict=True))
    # The real-recording transcripts do not depend on the checkpoint: cache them per item list.
    real_cache = Path(args.test_jsonl).with_suffix(f".asr_real_{args.asr.split('/')[-1]}_{args.n_per_set}.json")
    if real_cache.exists():
        real = json.load(open(real_cache))
    else:
        real = transcribe([c["ref"] for c in items])
        json.dump(real, open(real_cache, "w"), ensure_ascii=False)
    for i, c in enumerate(items):
        c["ref_norm"] = normalize(c["ref_text"])
        c["hyp"] = hyps.get(i, "")
        c["hyp_real"] = real[i]
        c["silent"] = int(i not in hyps)
        if i in hyps:
            c["wav"] = gens[i]
            c["sim"] = float(torch.cosine_similarity(embed(gens[i]), embed(c["ref"]), dim=0))
            if utmos is not None:
                with torch.no_grad():
                    c["utmos"] = float(utmos.score(load_16k(gens[i], device)[None, None]))

    def mean(xs):
        xs = list(xs)
        return round(sum(xs) / len(xs), 4) if xs else None

    results = {"step": step, "cfg": args.cfg, "temp": args.temp, "use_ema": args.use_ema}
    for name in ("hindi", "hinglish"):
        rows = [c for c in items if c["set"] == name]
        refs = [c["ref_norm"] for c in rows]
        results[name] = {
            "n": len(rows),
            "speakers": len({c["speaker"] for c in rows}),
            "wer": round(jiwer.wer(refs, [c["hyp"] for c in rows]), 4),
            "cer": round(jiwer.cer(refs, [c["hyp"] for c in rows]), 4),
            "wer_real_audio": round(jiwer.wer(refs, [c["hyp_real"] for c in rows]), 4),
            "cer_real_audio": round(jiwer.cer(refs, [c["hyp_real"] for c in rows]), 4),
            "sim": mean(c["sim"] for c in rows if "sim" in c),
            "utmos": mean(c["utmos"] for c in rows if "utmos" in c),
            "silent": sum(c["silent"] for c in rows),
            "no_eos": sum(c["no_eos"] for c in rows),
        }
    json.dump(results, open(out_dir / "results.json", "w"), indent=1, ensure_ascii=False)
    with open(out_dir / "items.jsonl", "w") as f:
        for c in items:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
