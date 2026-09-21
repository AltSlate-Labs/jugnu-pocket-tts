"""Generate the demo-page sample set: 3 models x 2 clean LibriVox reader voices x a fixed sentence list,
plus a clean-vs-degraded reference pair. Run from the pocket-tts checkout on a GPU."""
import json, sys
from pathlib import Path
import numpy as np, sphn, torch
sys.path.insert(0, str(Path.home() / "hindi-tts"))
import make_demo

T = Path.home() / "hindi-tts"
OUT = T / "page_samples"; OUT.mkdir(exist_ok=True)
SENT = {
    "hi1": "नमस्ते, मेरा नाम जुगनू है और मैं आपकी मदद करने के लिए यहाँ हूँ",
    "hi2": "क्या आप मुझे बता सकते हैं कि रेलवे स्टेशन यहाँ से कितनी दूर है",
    "hi_long": "आज सुबह से ही मौसम बहुत सुहावना है, हल्की हल्की बारिश हो रही है, और मैंने सोचा कि क्यों न शाम को सब दोस्तों के साथ चाय पर मिला जाए",
    "hi_num": "आपकी ट्रेन शाम सात बजकर पैंतालीस मिनट पर प्लेटफॉर्म नंबर तीन से रवाना होगी",
    "mix1": "कल की meeting तीन बजे reschedule कर देना और updated link मुझे भेज देना",
    "mix2": "आपका order confirm हो गया है और delivery कल शाम तक हो जाएगी",
    "mix_heavy": "मैंने अपना laptop restart किया, फिर software update install किया, लेकिन wifi अभी भी connect नहीं हो रहा",
    "en1": "Hello, thank you for calling. How can I help you today?",
    "en2": "The quick brown fox jumps over the lazy dog.",
}
MODELS = {
    "teacher-24l": (T / "runs/scratch_v0", T / "final_v0/scratch/checkpoint_00115000.pt", 2.0),
    "base-12l": (T / "runs/base12_v0", T / "final_v0/base12_00090000.pt", 1.0),
    "lite-6l": (T / "runs/lite6_v0", T / "final_v0/lite6_00112500.pt", 1.0),
}
# Degraded copy of the clean low reader: telephone band-limit + noise, to show what the prompt's acoustics do.
w, sr = sphn.read(str(T / "voices_clean/reader_low.wav")); w = w.mean(0)
w8 = sphn.resample(sphn.resample(w, src_sample_rate=sr, dst_sample_rate=8000), src_sample_rate=8000, dst_sample_rate=sr)
rng = np.random.default_rng(0); noisy = w8 + rng.normal(0, 0.02, len(w8)).astype(np.float32)
vdir = T / "voices_page"; vdir.mkdir(exist_ok=True)
for f in (T / "voices_clean").glob("*.wav"): (vdir / f.name).write_bytes(f.read_bytes())
sphn.write_wav(str(vdir / "reader_low_degraded.wav"), noisy.astype(np.float32), int(sr))

make_demo.SENTENCES = SENT; make_demo.VOICES = vdir
from training.eval.librispeech import latents_to_wav, load_mono, load_run
device = torch.device("cuda")
for name, (run_dir, ckpt, cfg) in MODELS.items():
    model, mimi, step = load_run(str(run_dir), device, use_ema=True, checkpoint=str(ckpt))
    torch.manual_seed(0); enc = model.flow_lm.conditioner.tokenizer.sp.encode
    (OUT / name).mkdir(exist_ok=True)
    for voice in sorted(vdir.glob("*.wav")):
        keys = list(SENT) if "degraded" not in voice.stem else ["hi1", "mix1", "en1"]
        wav = load_mono(str(voice), mimi.sample_rate)[: int(10 * mimi.sample_rate)]
        with torch.no_grad():
            lat = mimi.encode_to_latent(wav[None, None].to(device))[0]
            outs = model.generate([torch.tensor(enc(SENT[k]), dtype=torch.long) for k in keys], [lat] * len(keys),
                                  max_frames=int(30 * mimi.frame_rate), temp=0.3, n_steps=1, cfg_coef=cfg, eos_threshold=-1.0)
        for k, l in zip(keys, outs):
            a = latents_to_wav(mimi, l, device)
            if a is not None: sphn.write_wav(str(OUT / name / f"{voice.stem}__{k}.wav"), a.cpu().numpy(), int(mimi.sample_rate))
    print(name, "step", step, "clips", len(list((OUT / name).glob("*.wav"))))
    del model; torch.cuda.empty_cache()
json.dump(SENT, open(OUT / "sentences.json", "w"), ensure_ascii=False, indent=1)
