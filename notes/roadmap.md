# Hindi–English–Hinglish TTS — roadmap / phased to-do

Living list of what comes after the running work. Plan: [plan.md](plan.md) · decisions: [decisions.md](decisions.md)
· runs: [experiments.md](experiments.md).

## Now — v0 family on the frozen Kyutai codec (in progress)

- [x] Arm A `scratch_v0` ran to 400k (best at ~100–115k); Arm B `ftlang_v0` stopped at 115k; benchmarked every 25k steps.
- [x] Teacher picked (115k scratch checkpoint, D21); Base (12L) and Lite (6L) distilled to 200k; benchmark all three on quality, cloning, latency/memory.
- [ ] Remaining plan §8 tests: cross-language cloning, long text / names / numbers, 5 s vs 10 s references, Lite on CPU.
- [x] First human listening result: an unseen colleague's voice, "70% there, not an exact clone" (E8). Full listening pass still open.

## Published 2026-09-21 — v0 release

Create a **public repo `AltSlate-Labs/jugnu-pocket-tts`** (HF: `altslate/jugnu-pocket-tts`, gated) and push what we have: code, training recipe,
samples, roadmap, license. **Only after the results are out** (final benchmarks for the v0 family) — not before.
- [x] Repo contents: `hindi-tts/` scripts + configs, recipe write-up (data filters, Hinglish gloss swap, alignment,
      tokenizer, two-arm training, eval protocol, codec gate), `decisions.md`, `experiments.md` (including the
      zero-encoder failure), `roadmap.md`, benchmark tables, audio samples.
- [x] GitHub Pages demo site from `docs/` (audio player grid with clean-voice samples, benchmarks, how-to, limitations, acknowledgements).
- [x] README per house style: status badges + embedded diagrams (pipeline, model family).
- [x] License (code MIT; weights + tokenizer CC BY 4.0): MIT for our code (matches upstream pocket-tts; keep their notice where their code is referenced);
      `licenses.md` / NOTICE with CC BY 4.0 attributions — IndicVoices (AI4Bharat), HiFiTTS-2 (NVIDIA/LibriVox),
      HiACC (Singh, Singh & Kadyan 2025), Kyutai Pocket TTS / Mimi.
- [x] Decided with the user (D22–D28): repo name and visibility of model weights; whether samples may use
      the real-speaker presets (dataset adults, an 11-year-old from HiACC) or only a consented/synthetic-safe voice;
      whether to publish cloning-capable checkpoints at all vs preset-only voice states; re-read the accepted
      `kyutai/pocket-tts` terms for redistribution/use conditions; commit attribution rule for the public repo.
- [x] Python API `jugnu_tts` (clone / speak / stream / save voices), pip-installable from the repo.
- [x] Students finished 200k; benchmarked; published 90k / 112.5k checkpoints kept (D29); tables updated.
- [ ] Never push: dataset audio, Kyutai's gated weights, HF tokens, box address/keys.

## Next — own codec (deferred until v0 training completes; user decision 2026-09-19)

Why: v0's codec is Kyutai's pretrained Mimi, so v0 is not "all deployed weights trained in-project"; and the codec
gate measured it as lossier on our Hindi (23.8% ASR drift) than on English (1.0%) — experiments E3.
Kyutai released codec weights but no codec training code.

**Phase C0 — scoping (no GPU).**
- [ ] Audit open codec-training frameworks against the Pocket codec architecture (24 kHz, 12.5 fps, continuous
      32-dim bottleneck, encoder/decoder Transformers): `stable-audio-tools`, AudioCraft (EnCodec), DAC. Licenses.
- [ ] Choose the semantic-distillation target for Hindi/English (WavLM is English-centric; candidates w2v-BERT 2.0,
      IndicWav2Vec). Record it as pretrained *supervision* per plan §3.
- [ ] Wideband audio inventory. 16 kHz IndicVoices cannot teach a 24 kHz codec anything above 8 kHz.
      **Blocked on user:** accept terms for `ai4bharat/indicvoices_r` (48 kHz, 1,704 h) and `ai4bharat/Rasa`.
- [ ] Write `codec-plan.md`: architecture, losses, data, compute estimate, gates.

**Phase C1 — control: fine-tune Kyutai's codec on Indian speech** (~1–2 days GPU, estimate).
- [ ] Shows how much of the Hindi reconstruction loss is fixable, and sets the bar for C2. Not from-scratch.

**Phase C2 — codec from random init** (~3–6 days on 4 GPUs, estimate).
- [ ] Same architecture and losses as C1, random init, our audio pool.

**Phase C3 — gates (both must pass before any generator uses a new codec).**
- [ ] Reconstruction: `check_codec.py` — beat 23.8% Hindi drift, hold ~1% English; plus plan §3 blind listening
      (Hindi phonetics, switch points, speaker identity, pauses/breaths, artifacts, decoder runtime).
- [ ] Modellability: small generator, ~25k steps, same data, new latents vs Mimi latents → compare WER.

**Phase C4 — adopt.**
- [ ] Freeze codec (checkpoint, latent dim, frame rate, normalization stats); recompute latents; rerun Arm A's
      recipe; re-distill Base and Lite. This is the fully in-project family.

## Later — data

- [ ] Natural clause-level Hinglish and Indian English recordings from consented bilingual speakers (plan §4).
- [ ] Consented voice talent for the shipped male / female / kid presets (replaces dataset speakers).
- [ ] Child speech for training: HiACC children train/val (test split stays held out), Samrómur Children.
- [ ] `licenses.md` dependency record (datasets, pretrained tools, attribution text).
