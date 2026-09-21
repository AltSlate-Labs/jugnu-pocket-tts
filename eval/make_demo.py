"""Generate listening demos: every preset voice x a fixed Hindi / Hinglish / English sentence set.

Voices are reference clips in ~/hindi-tts/voices/<name>.(flac|wav). `pick-voices` seeds male
and female from held-out test speakers (never seen in training); a kid clip must be added by hand.

    python make_demo.py pick-voices <hi_test_aligned.jsonl>
    python make_demo.py generate <run_dir> [--checkpoint ckpt.pt]   # run from the pocket-tts checkout
"""

import argparse
import json
import shutil
from pathlib import Path

VOICES = Path.home() / "hindi-tts" / "voices"
SENTENCES = {
    "hi1": "नमस्ते, मेरा नाम जुगनू है और मैं आपकी मदद करने के लिए यहाँ हूँ",
    "hi2": "आज मौसम बहुत अच्छा है, चलिए शाम को पार्क में टहलने चलते हैं",
    "hi3": "क्या आप मुझे बता सकते हैं कि रेलवे स्टेशन यहाँ से कितनी दूर है",
    "mix1": "कल की meeting तीन बजे reschedule कर देना और updated link मुझे भेज देना",
    "mix2": "मैंने अपना phone charge पर लगाया था लेकिन battery अभी भी low है",
    "mix3": "आपका order confirm हो गया है और delivery कल शाम तक हो जाएगी",
    "en1": "Hello, thank you for calling. How can I help you today?",
    "en2": "The quick brown fox jumps over the lazy dog.",
}


def pick_voices(test_jsonl: str):
    VOICES.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(test_jsonl)]
    for gender, name in (("Male", "male"), ("Female", "female")):
        # A read, mid-length clip from a young-adult speaker: the cleanest reference available.
        cands = [
            r for r in rows
            if r["gender"] == gender and r["scenario"] == "Read" and 8 <= r["duration"] <= 14
            and r["age_group"] in ("18-30", "30-45")
        ]  # fmt: skip
        best = max(cands, key=lambda r: len(r["words"]) / r["duration"])
        shutil.copy(best["path"], VOICES / f"{name}.flac")
        print(name, best["speaker"], best["age_group"], best["duration"], best["transcript"][:80])


def generate(run_dir: str, checkpoint: str | None, cfg: float):
    import sphn
    import torch
    from training.eval.librispeech import latents_to_wav, load_mono, load_run

    device = torch.device("cuda")
    model, mimi, step = load_run(run_dir, device, use_ema=True, checkpoint=checkpoint)
    torch.manual_seed(0)
    out = Path(run_dir) / "demo" / f"step{step:08d}"
    out.mkdir(parents=True, exist_ok=True)
    encode = model.flow_lm.conditioner.tokenizer.sp.encode
    for voice in sorted(VOICES.iterdir()):
        wav = load_mono(str(voice), mimi.sample_rate)[: int(10 * mimi.sample_rate)]
        with torch.no_grad():
            latent = mimi.encode_to_latent(wav[None, None].to(device))[0]
            outs = model.generate(
                [torch.tensor(encode(t), dtype=torch.long) for t in SENTENCES.values()],
                [latent] * len(SENTENCES),
                max_frames=int(20 * mimi.frame_rate), temp=0.3, n_steps=1, cfg_coef=cfg,
                eos_threshold=-1.0,
            )  # fmt: skip
        for key, lat in zip(SENTENCES, outs, strict=True):
            audio = latents_to_wav(mimi, lat, device)
            if audio is not None:
                sphn.write_wav(str(out / f"{voice.stem}_{key}.wav"), audio.cpu().numpy(), int(mimi.sample_rate))
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pick-voices", "generate"])
    ap.add_argument("path")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--cfg", type=float, default=2.0)
    a = ap.parse_args()
    pick_voices(a.path) if a.cmd == "pick-voices" else generate(a.path, a.checkpoint, a.cfg)
