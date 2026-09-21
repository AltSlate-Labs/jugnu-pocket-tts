# Hindi–English–Hinglish TTS — experiments log

Box: 4× RTX PRO 4500 Blackwell 32 GB. Workspace `~/hindi-tts/` (upstream clone in `~/hindi-tts/pocket-tts`, pinned
per decisions D1). Scripts live in this repo under `hindi-tts/`. Decisions: [decisions.md](decisions.md).

## E0 — environment + access audit (2026-09-19)

- `uv sync` in the pinned clone → torch 2.13.0+cu130, CUDA available, capability (12, 0). Works on Blackwell as-is.
- Dataset access with the box's HF token: `ai4bharat/indicvoices_r` **403**, `ai4bharat/Rasa` **403**,
  `ai4bharat/indicwav2vec-hindi` **gated**, `SPRINGLab/IndicTTS-Hindi` OK, IndicVoices Hindi already on disk
  (83 shards, 16 kHz mono).
- IndicVoices Hindi survey (every 8th shard, 59,716 utt / 109.5 h): all `decision=excellent`; flags — noise_persistent
  22%, noise_intermittent 13%, chatter 6%, echo 3%, low_volume 2%. Gender/age spread is even (M/F × 18–30 … 60+);
  no under-18 speakers. ~6% of rows carry `[English]` glosses.

## E1 — data preparation (2026-09-19, done)

| Step | Result |
| --- | --- |
| HiFiTTS-2 200 h + Kyutai published alignments (`prepare_data.py --hours 200`) | 65,733 train utt, 412 valid utt; 1.1 h lost to dead LibriVox links |
| IndicVoices Hindi filter (`extract_iv_hindi.py`) | 450,690 seen → **193,039 kept, 453.5 h** (16 kHz flac, 27 GB); 67,290 utt (35%) carry usable `[English]` glosses |
| Speaker-disjoint split (`build_manifests.py split`) | train 190,132 utt / 446.7 h / 2,143 spk · valid 1,563 utt / 3.8 h / 20 spk · test 1,344 utt / 3.0 h / 20 spk |
| Hindi forced alignment (vakyansh wav2vec2, 4 GPUs, ~8.5 min) | 190,126 / 190,132 train aligned (6 skipped); valid/test 100%. Check on 20k rows: 0 rows with an unaligned word, trailing silence p50 0.20 s / p90 0.54 s, 2.6 words/s |
| Merge + Hinglish swap (`build_manifests.py finalize`) | `data/v0/train_aligned.jsonl` = 255,859 utt (Hindi 190k + English 66k); ~half of glossed Hindi utterances use Latin spellings |
| Tokenizer | SentencePiece BPE 4000 over all train transcripts → `data/v0/tokenizer.model` |

Data mix by hours: Hindi ≈ 447 h (69%), English ≈ 200 h (31%); word-level Hinglish is inside the Hindi share.
No Indian English, no clause-level Hinglish, no child speech (see decisions D4, D5, D9).

Access finding: `kyutai/pocket-tts` (voice-cloning weights) is gated for this account; the ungated
`kyutai/pocket-tts-without-voice-cloning` 24L file contains the full Mimi (encoder + decoder) and FlowLM, so
`configs/hinglish_24l.yaml` points `weights_path` there.

## E2 — v0 teachers (24L, frozen Mimi), two arms — running

Launched 2026-09-19 via `launch_train.sh`. Arm A `scratch_v0` on GPUs 0,1; Arm B `ftlang_v0` on GPUs 2,3
(decision D11). Both: batch 16 × 2 GPUs × accum 2 = 64, lr 2e-4, LSD flow, CFG dropout 0.2/0.2.

Throughput: latent precompute ~13 min (both arms encoded the same cache concurrently — safe, upstream uses atomic
renames); then **~3.4 optimizer steps/s per arm, ~27 GB/GPU**. 50k steps ≈ 4 h, 400k ≈ 33 h.

### Eval protocol (`eval_tts.py` + `score_indicconf.py`, automated by `watch_eval.sh` every 25k steps)

- Items: held-out test speakers only (20 spk, never in training). Prompt = first 5 s of a *different* utterance of
  the same speaker; target text 3–12 s utterances. 150 Hindi items (pure Devanagari) + 150 Hinglish items
  (mixed-script text built from the glosses). EMA weights, temp 0.3, cfg 2.0, 1 LSD step, eos-threshold −1.
- Intelligibility: **IndicConformer-600M RNNT (primary)** and whisper-large-v3-turbo (secondary), both scored
  against the Devanagari transcript with light normalization (NFC, nukta dropped, chandrabindu→anusvara,
  punctuation stripped). Every WER is reported next to the same ASR's WER on the *real* target recordings.
- Speaker similarity: wavlm-base-plus-sv cosine, generated vs the real target recording. UTMOS where installed
  (English-trained; treat as a rough signal on Hindi).
- `silent` / `no_eos` counts = empty generations / generations that ran to the 30 s cap.

ASR floor on the real recordings (20-item smoke set): whisper-turbo WER 39% (Hindi) / 41% (Hinglish) — too weak to
judge this speech; IndicConformer WER 15% / 14%, CER 7% / 6%. Hence IndicConformer is primary.

Eval engineering notes: whisper on CPU ran single-threaded (unusable); whisper-large-v3 does not fit beside a
27 GB training job, turbo with greedy decoding and batch 4 does; clips are capped at 29 s before ASR because the HF
pipeline cannot batch long-form audio.

### Results

| Arm | Step | Set | IC WER | IC CER | (real-audio IC WER) | Whisper WER | Sim | UTMOS | silent / no_eos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ftlang (raw weights, 20-item smoke) | 10k | Hindi | 100% | 99% | 15% | 99% | 0.19 | 1.33 | 0 / 18 of 20 |
| ftlang (raw weights, 20-item smoke) | 10k | Hinglish | 100% | 100% | 14% | 99% | 0.21 | 1.33 | 0 / 16 of 20 |

At 10k steps the model is not speaking yet (expected: upstream sees WER start to fall at ~15k).

### E2 outcome — INVALID, both arms stopped at ~63k steps (2026-09-19)

25k/50k benchmarks (150 items per set, EMA): IndicConformer WER 100% on Hindi and Hinglish in both arms, speaker
sim 0.17–0.19, UTMOS 1.34–1.37, 96–102 of 150 generations running to the 30 s cap. The trainer's own step-60k
samples (including the English sentence, in the arm that started from a working English model) transcribe as
silence/"Music". Training loss was falling normally the whole time.

**Root cause: the codec encoder in the ungated weights file is zeroed.** `kyutai/pocket-tts-without-voice-cloning`
contains `mimi.encoder.*` tensors, but every encoder weight is exactly 0 (that is how the voice-cloning capability
is withheld). Encode→decode of a real HiFiTTS clip gives latents with mean 0 / std 0 and a reconstruction that
transcribes as "." (original transcribes perfectly). Every precomputed training latent and every voice prompt was
therefore all-zeros: the models learned to predict a constant. ~5.5 h × 4 GPUs lost.

What is still valid: the whole data pipeline (manifests, alignments, tokenizer), the eval harness, presets.
What is not: `data/v0/latents/` (keyed by encoder hash, so a working encoder gets a fresh cache automatically) and
everything under `runs/*_v0/`.

Process failure: plan §3 defines an autoencoder reconstruction gate *before* generator training. I skipped it
because the codec was "released and frozen", and I took the presence of encoder tensors as proof they worked.
The gate is now a script (`hindi-tts/check_codec.py`) that must pass before any launch.

## E3 — codec reconstruction gate on the gated Kyutai weights (2026-09-19)

User accepted the `kyutai/pocket-tts` terms; `configs/hinglish_24l.yaml` now points at the gated 24L file.
`check_codec.py` on 40 validation clips (whisper-turbo, language forced, original vs encode→decode transcript):

| | latent std (min) | ASR drift original→reconstruction (WER) | limit | |
| --- | --- | --- | --- | --- |
| English (22 clips, HiFiTTS-2 44.1 kHz) | 0.85 | **1.0%** | 10% | pass |
| Hindi (18 clips, IndicVoices 16 kHz) | 0.85 | **23.8%** | 35% | pass |

First run of the gate failed at 47.6% pooled drift: on auto-detect whisper flipped several Hindi clips to Urdu
script or translated them to English. With the language forced, English is lossless to the ASR, and Hindi still
shows real losses on some clips (e.g. `प्रकार`→`ततार`, `राष्ट्रीय आय`→`राज की आएका`) on top of judge noise.
**Finding for plan §3:** the English-trained Mimi is measurably lossier on our 16 kHz field-recorded Hindi. It sets
a ceiling on v0 Hindi intelligibility and is the first concrete argument for a codec trained on Indian speech.
Not yet separated: codec language bias vs the 16 kHz / noisy source audio (re-test on 48 kHz IndicVoices-R Hindi).

## E4 — v0 teachers relaunched on real latents (2026-09-19) — running

Same configs, data, tokenizer and seeds as E2; invalid runs and the zero-latent cache deleted (logs kept in
`~/hindi-tts/invalid_zero_latents/`). `launch_train.sh` now runs the codec gate before launching.
Early sanity check added: the trainer's own step-5k/10k samples are transcribed before trusting the run.

### E4 early sanity check — trainer samples at 5k / 10k steps (whisper-turbo transcripts, raw weights)

Prompts: `आज मौसम बहुत अच्छा है और हम सब बाहर घूमने जा रहे हैं` · `कल की meeting तीन बजे reschedule कर देना और updated
link मुझे भेज देना` · `The quick brown fox jumps over the lazy dog.`

| Arm | Step | Hindi | Hinglish | English |
| --- | --- | --- | --- | --- |
| scratch | 5k | unintelligible | unintelligible | unintelligible |
| scratch | 10k | 2 word errors (`बाहर घूमने`→`भार भूमिये`) | meeting ✓, reschedule ✗, updated/link garbled | `The good quick round folks jumps over the lazy dough` |
| ftlang | 5k | 1 word error (`घूमने`→`भूमने`) | meeting/updated/link ✓, reschedule ✗ | exact |
| ftlang | 10k | **exact** | meeting/updated/link ✓, reschedule ✗ (`रेस्ट्राल`), `भेज`→`भीज` | exact |

The run is valid this time. Inheriting the English teacher (Arm B) is ~2× ahead of scratch at equal steps, in line
with upstream's Czech result; scratch is itself ahead of upstream's "WER starts dropping ~15k" timeline.
First three-voice demo (male / female / kid × 8 sentences, ftlang step 10k EMA) copied to
`~/Downloads/hindi-tts-v0-samples/` on the Mac — not yet listened to by a human.

### E4 benchmark — step 25k (EMA, 150 items per set, 20 held-out speakers, cfg 2.0, temp 0.3)

| Arm | Set | IC WER | IC CER | real-audio IC WER / CER | Whisper-turbo WER | Speaker sim | UTMOS | silent / no_eos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | Hindi | 9.7% | 3.6% | 14.5% / 6.2% | 29.4% | 0.913 | 2.03 | 0 / 0 |
| scratch | Hinglish | 14.0% | 6.7% | 15.7% / 6.6% | 36.0% | 0.905 | 1.99 | 0 / 0 |
| ftlang | Hindi | **9.1%** | **3.1%** | 14.5% / 6.2% | 28.9% | **0.921** | **2.62** | 0 / 0 |
| ftlang | Hinglish | **12.7%** | **5.6%** | 15.7% / 6.6% | 31.9% | **0.920** | **2.48** | 0 / 0 |

Speaker similarity by gender: scratch F 0.903 / M 0.914; ftlang F 0.922 / M 0.919 (≈150 items each).

Reading:
- Both arms are intelligible on unseen speakers at 25k steps, with no empty or runaway generations.
- Synthetic WER is *below* the WER of the real recordings. That does not mean "better than human": the real clips
  are spontaneous, noisy field recordings, the TTS output is clean, and IndicConformer was trained on IndicVoices-style
  speech. Treat ~9–14% as "at the judge's floor"; differences between arms (≤1.3 pts) are within noise at n=150.
- Speaker similarity 0.90–0.92 is close to upstream's English teacher (0.93). Caveat: wavlm-base-plus-sv is
  English-trained and unvalidated on Hindi (plan §8), and prompt and target share recording conditions, which inflates it.
- The clear gap is acoustic quality: UTMOS 2.0 (scratch) vs 2.6 (ftlang) vs upstream's 4.3 at 400k. Expected — upstream
  says UTMOS only lifts at 150–200k steps; also UTMOS is English-trained and our Hindi audio is 16 kHz.
- Hinglish is 3–4 pts harder than Hindi in both arms.

### E4 benchmark — step 50k (same protocol and items as 25k)

| Arm | Set | IC WER | IC CER | Whisper-turbo WER | Speaker sim | UTMOS | silent / no_eos | Δ IC WER vs 25k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | Hindi | 10.0% | 3.7% | 34.7% | 0.911 | 2.11 | 0 / 0 | +0.3 |
| scratch | Hinglish | 13.8% | 6.8% | 42.8% | 0.908 | 2.06 | 0 / 0 | −0.2 |
| ftlang | Hindi | 8.9% | 3.2% | 27.7% | 0.917 | 2.60 | 0 / 0 | −0.2 |
| ftlang | Hinglish | 13.4% | 6.2% | 36.6% | 0.915 | 2.49 | 0 / 0 | +0.7 |

Reading: intelligibility and speaker similarity have plateaued (all changes within noise at n=150) — consistent
with upstream ("WER flat after ~50k; the rest is acoustic quality"). UTMOS is flat too (2.1 / 2.6), as expected
before the 150–200k transition. Whisper-turbo WER moved the wrong way for scratch (29→35%, 36→43%) while the primary
judge did not; whisper is the noisy secondary judge here, but watch it at 75k/100k.
Validation loss at 52.5k: scratch 0.117, ftlang 0.074.

### E4 benchmark — step 75k

| Arm | Set | IC WER | IC CER | Whisper-turbo WER | Speaker sim | UTMOS | silent / no_eos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | Hindi | 10.3% | 4.5% | 29.2% | 0.914 | 2.21 | 0 / 0 |
| scratch | Hinglish | 13.0% | 6.1% | 34.0% | 0.914 | 2.11 | 0 / 0 |
| ftlang | Hindi | 9.3% | 3.5% | 36.2% | 0.915 | 2.62 | 0 / 0 |
| ftlang | Hinglish | 12.9% | 5.7% | 35.2% | 0.908 | 2.42 | 0 / 0 |

Still on the plateau. Scratch's whisper WER came back down (35→29%, 43→34%) while ftlang's Hindi went up
(28→36%): the whisper numbers swing ±7 pts between checkpoints with no matching move in the primary judge, so the
50k scratch bump was judge noise, not a regression. Scratch UTMOS is creeping up (2.03→2.11→2.21); ftlang is flat.

### E4 benchmark — step 100k

| Arm | Set | IC WER | IC CER | Whisper-turbo WER | Speaker sim | UTMOS | silent / no_eos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | Hindi | 9.0% | 3.3% | 32.7% | 0.916 | 2.35 | 0 / 0 |
| scratch | Hinglish | 12.9% | 6.3% | 39.1% | 0.909 | 2.22 | 0 / 0 |
| ftlang | Hindi | 8.8% | 2.9% | 29.8% | 0.917 | 2.58 | 0 / 0 |
| ftlang | Hinglish | 12.1% | 5.0% | 40.2% | 0.910 | 2.45 | 0 / 0 |

Best IC WER/CER so far for both arms, still within plateau noise. Scratch has closed the Hindi WER gap to 0.2 pts and
its UTMOS keeps rising (2.03 → 2.11 → 2.21 → 2.35); ftlang UTMOS is flat at ~2.6 (2.62 → 2.60 → 2.62 → 2.58).
Validation loss rose 82.5k→100k in both arms (scratch 0.109→0.129, ftlang 0.079→0.093) with no matching benchmark
regression. UTMOS trend by step, scratch/ftlang Hindi: 25k 2.03/2.62 · 50k 2.11/2.60 · 75k 2.21/2.62 · 100k 2.35/2.58.

### E4 stopped at step 115k (2026-09-19 14:14 UTC) — user needed the GPUs

Both arms stopped cleanly after checkpoint 115,000 (`stop_at_115k.sh`). Upstream's trainer traps SIGTERM and wrote
a final checkpoint + optimizer pair at the exact stop step (scratch 115,192; ftlang 115,511), so both runs are
resumable by relaunching the same config. Archived in `~/hindi-tts/final_v0/` (17 GB): per arm
`checkpoint_00115000.pt` (benchmarked, with EMA), `model.safetensors`, `args.yaml`, the resume pair; plus
`tokenizer.model` and `hinglish_24l.yaml`. Progress: scratch 29% of 400k, ftlang 46% of 250k. The 150–200k
acoustic-quality transition was **not reached**.

### v0 result — step 115k (EMA, 150 items per set, 20 held-out speakers, cfg 2.0, temp 0.3)

| Arm | Set | IC WER | IC CER | real-audio IC WER / CER | Whisper-turbo WER | Speaker sim | UTMOS | silent / no_eos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | Hindi | 9.1% | 3.3% | 14.5% / 6.2% | 34.5% | 0.915 | 2.48 | 0 / 0 |
| scratch | Hinglish | 13.3% | 6.7% | 15.7% / 6.6% | 39.4% | 0.907 | 2.30 | 0 / 0 |
| ftlang | Hindi | 9.1% | 3.3% | 14.5% / 6.2% | 29.6% | 0.920 | 2.59 | 0 / 0 |
| ftlang | Hinglish | 12.2% | 5.1% | 15.7% / 6.6% | 43.3% | 0.913 | 2.44 | 0 / 0 |

Trajectory (Hindi IC WER / UTMOS): scratch 25k 9.7/2.03 · 50k 10.0/2.11 · 75k 10.3/2.21 · 100k 9.0/2.35 · 115k 9.1/2.48;
ftlang 25k 9.1/2.62 · 50k 8.9/2.60 · 75k 9.3/2.62 · 100k 8.8/2.58 · 115k 9.1/2.59.

Conclusions for v0:
- From-scratch generator (Arm A) matches the English-initialised one (Arm B) on Hindi intelligibility (9.1% = 9.1%)
  and speaker similarity (0.915 vs 0.920) by 115k steps; Arm B keeps ~1 pt on Hinglish WER and ~0.1–0.15 UTMOS.
- Scratch UTMOS rises steadily (2.03→2.48) and is converging on ftlang's flat ~2.6; neither reached the upstream
  transition, so naturalness is the unfinished part. Both are far below upstream's 4.3 (English, 400k, 44 kHz data).
- Not done: English benchmark (LibriSpeech protocol), cross-language cloning, long text / names / numbers,
  5 s vs 10 s references, Base/Lite distillation, latency / CPU tests, any human listening.
- Demos (male / female / kid × 8 sentences, both arms, steps 55k and 115k): `~/Downloads/hindi-tts-v0-samples/`.

## E5 — Arm A resumed (2026-09-19 15:41 UTC) — running

`resume_scratch.sh`: GPUs 2,3, auto-resumed from `checkpoint_00115192.pt` (log: "resumed from … (step 115192)"),
lr 1.62e-4 continuing the cosine, ~3.0 steps/s, ~21 GB/GPU. Benchmarks continue every 25k steps (next: 125k).

### E5 benchmark — scratch, step 125k

| Set | IC WER | IC CER | Whisper-turbo WER | Speaker sim | UTMOS | silent / no_eos |
| --- | --- | --- | --- | --- | --- | --- |
| Hindi | 9.4% | 4.1% | 31.1% | 0.914 | 2.50 | 0 / 0 |
| Hinglish | 13.5% | 6.3% | 35.6% | 0.909 | 2.36 | 0 / 0 |

WER / similarity still on the plateau; UTMOS continues its slow climb (Hindi 2.03 → 2.11 → 2.21 → 2.35 → 2.48 →
2.50). Resume was seamless: no discontinuity versus the 115k numbers.

### E5 benchmark — scratch, step 150k

| Set | IC WER | IC CER | Whisper-turbo WER | Speaker sim | UTMOS | silent / no_eos |
| --- | --- | --- | --- | --- | --- | --- |
| Hindi | 10.1% | 3.7% | 32.9% | 0.914 | 2.56 | 0 / 0 |
| Hinglish | 12.4% | 6.0% | 35.7% | 0.916 | 2.37 | 0 / 0 |

Start of upstream's 150–200k transition window: no jump. UTMOS keeps the same slow slope (Hindi 2.50 → 2.56,
Hinglish 2.36 → 2.37); WER and similarity unchanged within noise. Scratch has now reached Arm B's UTMOS level (~2.6).

### E5 benchmarks — scratch, steps 175k and 200k

| Step | Set | IC WER | IC CER | Whisper-turbo WER | Speaker sim | UTMOS | silent / no_eos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 175k | Hindi | 10.8% | 4.7% | 30.5% | 0.916 | 2.55 | 0 / 0 |
| 175k | Hinglish | 13.3% | 6.3% | 48.8% | 0.913 | 2.41 | 0 / 0 |
| 200k | Hindi | 9.8% | 4.1% | 35.7% | 0.913 | 2.59 | 0 / 0 |
| 200k | Hinglish | 13.7% | 6.3% | 38.1% | 0.910 | 2.44 | 0 / 0 |

Upstream's 150–200k transition window has passed with **no UTMOS jump**: Hindi 2.50 → 2.56 → 2.55 → 2.59.

## E6 — why UTMOS sits at ~2.6: it is the reference audio, not the model (2026-09-20)

Same UTMOS model, scored on real audio:

| Audio | UTMOS |
| --- | --- |
| Real held-out IndicVoices recordings — Hindi items (n=150) | **2.04** |
| Real held-out IndicVoices recordings — Hinglish items (n=150) | **1.98** |
| Generated speech for those items (scratch 200k) | 2.59 / 2.44 |
| Real HiFiTTS-2 English validation recordings (n=60) | 3.60 |
| Preset reference clips: male / female / kid | 1.67 / 1.70 / 2.56 |

The model already scores *above* the recordings it is asked to imitate. A cloning TTS reproduces the acoustics of
its prompt, and the benchmark prompts (and the male/female presets) are noisy 16 kHz field recordings.

Direct test — scratch step 202.5k, same 8 demo sentences, four reference voices (UTMOS by output language):

| Reference voice | English | Hindi | Hinglish |
| --- | --- | --- | --- |
| **clean studio English reader (HiFiTTS-2 valid speaker 2104, 10 s)** | **3.87** | **3.65** | **3.74** |
| kid (HiACC CH26) | 3.04 | 2.20 | 2.35 |
| female (IndicVoices test speaker) | 2.84 | 2.44 | 2.40 |
| male (IndicVoices test speaker) | 2.29 | 1.89 | 1.73 |

With a clean reference the same checkpoint speaks Hindi at UTMOS 3.65 and Hinglish at 3.74 — cross-lingually, from
an English-only reference. IndicConformer transcripts of those six clips: 3 of 6 exact, the rest with 1–3 word
errors (`नमस्ते`→`नष्ट`, `पार्क में टहलने`→`पार्ट में ठहलने`, `तीन बजे reschedule`→`थी मुझे रेस्टो`). n is tiny; a proper
cross-lingual set is still owed (plan §8).

Conclusions:
1. The "missing quality transition" is mostly a measurement artefact of noisy prompts; do not judge teacher
   naturalness on IndicVoices-prompted UTMOS. Add a clean-reference track to the benchmark.
2. **Preset voices should be clean studio recordings.** The current male/female presets (UTMOS 1.7) are the worst
   possible showcase; the consented-voice-talent item in the roadmap is now also the biggest quality lever.
3. 48 kHz clean Hindi (IndicVoices-R, Rasa) matters for Hindi timbre in Hindi voices, less for intelligibility.

### E5 benchmark — scratch, step 225k

| Set | IC WER | IC CER | Whisper-turbo WER | Speaker sim | UTMOS | silent / no_eos |
| --- | --- | --- | --- | --- | --- | --- |
| Hindi | 10.6% | 4.3% | 36.0% | 0.914 | 2.60 | 0 / 0 |
| Hinglish | 14.5% | 7.1% | 43.6% | 0.915 | 2.39 | 0 / 0 |

Plateau continues. Hinglish 14.5% is the highest IC WER of the run so far (range 12.4–14.5%), still within the
±1 pt checkpoint-to-checkpoint noise seen since 25k.

### E5 benchmark — scratch, step 250k

| Set | IC WER | IC CER | Whisper-turbo WER | Speaker sim | UTMOS | silent / no_eos |
| --- | --- | --- | --- | --- | --- | --- |
| Hindi | 10.9% | 4.4% | 40.3% | 0.914 | 2.58 | 0 / 0 |
| Hinglish | 13.7% | 6.3% | 37.4% | 0.913 | 2.41 | 0 / 0 |

Watch item: Hindi IC WER has drifted up since its 100k low — 9.0 (100k) · 9.1 · 9.4 · 10.1 · 10.8 · 9.8 · 10.6 ·
10.9 (250k). Each step is inside the ±1 pt noise at n=150, but the direction is consistent (~+1.5 pts over 150k
steps); CER moved 3.3 → 4.4. Hinglish shows no trend (12.4–14.5). Speaker similarity and UTMOS are flat. Not acted on:
the same 150 items and seed are used every time, so this is a real (small) change in the model, possibly the model
trading exact wording for the noisy training distribution's disfluent style. To resolve at the end of the run:
re-score 100k-vs-final on a larger item set (all ~1,300 test utterances) and sweep eos-threshold / cfg.

### E5 benchmarks — scratch, steps 275k and 300k: the Hindi drift is real

| Step | Set | IC WER | IC CER | Whisper-turbo WER | Speaker sim | UTMOS | silent / no_eos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 275k | Hindi | 12.0% | 5.4% | 32.6% | 0.916 | 2.59 | 0 / 0 |
| 275k | Hinglish | 14.4% | 6.7% | 45.4% | 0.915 | 2.42 | 0 / 0 |
| 300k | Hindi | 12.2% | 6.1% | 51.6% | 0.914 | 2.58 | 0 / 0 |
| 300k | Hinglish | 15.3% | 7.6% | 51.4% | 0.910 | 2.40 | 0 / 0 |

Error breakdown (IndicConformer, % of reference words):

| Step | Set | S | D | I | mean gen duration (real: 7.17 / 6.90 s) | items with WER > 50% | median item WER |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100k | Hindi | 7.0 | 1.5 | 0.5 | 5.36 s | 0 | 7.7% |
| 200k | Hindi | 7.2 | 1.7 | 0.9 | 5.49 s | 3 | 7.7% |
| 250k | Hindi | 8.1 | 1.8 | 1.1 | 5.60 s | 2 | 8.5% |
| 300k | Hindi | 7.6 | 3.5 | 1.1 | 5.91 s | 6 | 8.3% |
| 100k | Hinglish | 8.9 | 3.3 | 0.7 | 5.18 s | 7 | 10.0% |
| 300k | Hinglish | 10.4 | 3.8 | 1.1 | 5.72 s | 9 | 12.5% |

Reading: the typical item has barely moved (median 7.7 → 8.3%); substitutions are flat. The increase is a growing
**tail of generations the ASR cannot transcribe at all** (deletions 1.5 → 3.5%, items > 50% WER 0 → 6; e.g. two
items go from a perfect transcript at 100k to an empty one at 300k) plus slower, longer speech (5.36 → 5.91 s,
moving toward the real recordings' 7.17 s). Interpretation: with more training the model imitates the noisy,
hesitant field-recording *style* of its prompts more faithfully — better cloning of the distribution, worse
intelligibility on the hardest prompts. Speaker similarity and UTMOS are unchanged.
`checkpoint_00300000.pt` archived to `final_v0/scratch_300k/`. Follow-up running: sampling sweep on the 300k
checkpoint (cfg 3.0 / temp 0.3, and cfg 2.0 / temp 0.1) to see whether the tail is removable at inference.

### E5 follow-up — sampling sweep on the 300k checkpoint (same 300 items)

| Sampling | Set | IC WER | S / D / I | items > 50% WER | Speaker sim | UTMOS |
| --- | --- | --- | --- | --- | --- | --- |
| cfg 2.0, temp 0.3 (standard) | Hindi | 12.2% | 7.6 / 3.5 / 1.1 | 6 | 0.914 | 2.58 |
| cfg 2.0, temp 0.1 | Hindi | 11.8% | 8.6 / 2.1 / 1.1 | 6 | 0.914 | 2.60 |
| cfg 3.0, temp 0.3 | Hindi | 16.7% | 9.5 / 6.2 / 1.1 | 14 | 0.886 | 2.21 |
| cfg 2.0, temp 0.3 (standard) | Hinglish | 15.3% | 10.4 / 3.8 / 1.1 | 9 | 0.910 | 2.40 |
| cfg 2.0, temp 0.1 | Hinglish | 15.0% | 10.1 / 3.8 / 1.1 | 9 | 0.915 | 2.47 |
| cfg 3.0, temp 0.3 | Hinglish | 21.8% | 11.5 / 8.6 / 1.7 | 23 | 0.887 | 2.06 |

Lower temperature does not remove the unintelligible tail (same 6 / 9 bad items); stronger guidance makes
everything worse (WER, similarity and UTMOS). The tail is in the model, not the sampler. Running next: cfg 1.5 and
1.0. Teacher checkpoints 325k / 350k / 375k are now archived as they appear (`archive_ckpts.sh`).

Lower-guidance half of the sweep (300k checkpoint, temp 0.3):

| Sampling | Set | IC WER | S / D / I | items > 50% WER | Speaker sim | UTMOS | no_eos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cfg 1.5 | Hindi | 12.7% | 8.3 / 3.5 / 0.9 | 5 | 0.919 | 2.62 | 0 |
| cfg 1.5 | Hinglish | 16.0% | 11.1 / 3.9 / 0.9 | 13 | 0.914 | 2.46 | 0 |
| cfg 1.0 (no guidance) | Hindi | 21.8% | 10.6 / 10.3 / 0.9 | 19 | 0.904 | 2.39 | 5 |
| cfg 1.0 (no guidance) | Hinglish | 25.0% | 12.8 / 11.1 / 1.1 | 27 | 0.886 | 2.27 | 8 |

cfg 2.0 / temp 0.3 (upstream's default) is the best operating point on every axis; 1.5 is a near tie, 1.0 and 3.0
are clearly worse. **No sampling setting recovers the 100k-step Hindi WER (9.0%)** — the late-training tail is a
property of the weights. Practical consequence: for this data, the best teacher for intelligibility was ~100–150k
steps; longer training bought no UTMOS and cost ~3 pts of Hindi WER.

### E5 final — scratch teacher reached 400k (2026-09-20 19:54 UTC): late training made it worse

| Step | Hindi IC WER (S / D / I) | Hinglish IC WER (S / D / I) | items > 50% WER (Hi / Mix) | Speaker sim | UTMOS (Hi / Mix) |
| --- | --- | --- | --- | --- | --- |
| 100k | 9.0% (7.0 / 1.5 / 0.5) | 12.9% (8.9 / 3.3 / 0.7) | 0 / 7 | 0.916 / 0.909 | 2.35 / 2.22 |
| 115k | 9.1% | 13.3% | — | 0.915 / 0.907 | 2.48 / 2.30 |
| 300k | 12.2% (7.6 / 3.5 / 1.1) | 15.3% (10.4 / 3.8 / 1.1) | 6 / 9 | 0.914 / 0.910 | 2.58 / 2.40 |
| 350k | 14.6% (9.0 / 3.8 / 1.7) | 19.6% (11.1 / 6.9 / 1.6) | 11 / 17 | 0.914 / 0.909 | 2.53 / 2.40 |
| 375k | 15.4% (10.1 / 3.8 / 1.5) | 20.1% (12.0 / 6.6 / 1.4) | 10 / 19 | 0.913 / 0.907 | 2.54 / 2.38 |
| 400k | 14.3% (CER 7.1%) | 18.9% (CER 10.7%) | — | — | — |

From 300k on, substitutions rise as well as deletions, so it is no longer only a tail effect. The fully trained
teacher is ~5 pts worse on Hindi and ~6 pts worse on Hinglish than the 100–115k checkpoints, with no gain in
similarity or (prompt-capped) UTMOS. **Best v0 teacher = the archived 115k scratch checkpoint.** Upstream's "never
shorten the schedule" advice does not transfer to this noisy 16 kHz-dominated corpus. For the next teacher: cosine
to ~150k, or early-stop on benchmark WER; and fix the data (IndicVoices-R/Rasa, stricter filtering) first.
Archived teachers: `final_v0/scratch` (115k), `scratch_300k`, `_325k`, `_350k`, `_375k`, `_400k`.

## E7 — distillation: Base 12L + Lite 6L (2026-09-20/21)

- **Attempt 1 (chain, 19:59 UTC, teacher = 400k, batch 64 × 1 GPU): both crashed with CUDA OOM in the validation
  pass** — validation encodes raw audio through Mimi at the full batch, a single ~9 GB allocation on top of ~22 GB.
  Base died at its first validation (step 2,500, nothing saved); Lite at step 10,000 (last checkpoint 7,500). GPUs
  sat idle ~4 h until the next status check. My 40-step smoke test had validation disabled, so it could not catch
  this. Fixes: batch 32 × accum 2 (same effective 64), `expandable_segments`, and `supervise_student.sh`, which
  auto-resumes a crashed student from its latest checkpoint (up to 8 attempts).
- **Attempt 2 (00:24 UTC 21 Sep): teacher switched to the 115k checkpoint** (decision D21) before any meaningful
  progress was lost; attempt-1 artefacts kept in `~/hindi-tts/invalid_400k_teacher/`.
  Base 12L on GPU 2: 2.45 steps/s, 12.7 GB → 200k ≈ 22.7 h. Lite 6L on GPU 3: 3.2 steps/s, 7.7 GB → 200k ≈ 17.4 h.
  Benchmarks every 50k steps at cfg 1.0 (`watch_students.sh`).
- Tooling gotcha hit again: `pkill -f supervise_student` inside an SSH one-liner matched the SSH command itself
  (the pattern text appears in the command line), killing the session half-way. Multi-step process control now goes
  through a script file on the box.

### E7 results so far — students match the teacher (2026-09-21)

| Model | Step | Hindi IC WER / CER | Hinglish IC WER / CER | Speaker sim (Hi / Mix) | UTMOS (Hi / Mix) | silent / no_eos |
| --- | --- | --- | --- | --- | --- | --- |
| Teacher 24L (cfg 2.0) | 115k | 9.1% / 3.3% | 13.3% / 6.7% | 0.915 / 0.907 | 2.48 / 2.30 | 0 / 0 |
| Lite 6L (cfg 1.0) | 50k | 8.4% / 3.2% | 12.1% / 5.2% | 0.916 / 0.911 | 2.57 / 2.41 | 0 / 1 |
| Lite 6L (cfg 1.0) | 100k | 9.0% / 3.2% | 12.2% / 5.2% | 0.916 / 0.913 | 2.54 / 2.39 | 0 / 1 |
| Base 12L (cfg 1.0) | 60k | 9.0% / 3.2% | 12.1% / 5.5% | 0.916 / 0.908 | 2.56 / 2.39 | 0 / 0 |

Parameters (FlowLM only): teacher 316.0 M, Base 165.0 M, Lite 89.4 M. Base's 50k benchmark was lost: its trainer keeps
~31 GB reserved on GPU 2, the eval OOM'd three times and the checkpoint rotated away; all student evals now run on
GPU 3 and the 60k checkpoint was archived and benchmarked instead.

Demo-page sample set (`release/gen_page_samples.py`): 3 models × 2 clean LibriVox readers (speaker 2104, also in
the English training data; speaker 10644, unseen) × 9 sentences + a clean-vs-degraded reference pair. Every clip was
transcribed before publishing: students are near-exact on Hindi, numbers and heavy code-switching cross-lingually;
`reschedule` is mangled by all three models in every clip; the teacher is noisier than its students on these prompts.

Release staging: box `~/hindi-tts/hf_release/{teacher-24l,base-12l,lite-6l}` (FlowLM-only + `assemble.py`); local
repo `~/Workspace/jugnu-pocket-tts` (committed locally, not pushed) with README, NOTICE, figures and `docs/` page.
