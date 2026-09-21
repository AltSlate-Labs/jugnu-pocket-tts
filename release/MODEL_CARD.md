---
license: cc-by-4.0
language:
  - hi
  - en
pipeline_tag: text-to-speech
tags:
  - text-to-speech
  - hindi
  - hinglish
  - code-switching
  - voice-cloning
  - pocket-tts
  - calm
library_name: pocket-tts
base_model: kyutai/pocket-tts
extra_gated_prompt: >-
  Prohibited use: Use of this model must comply with all applicable laws and regulations and must not result in,
  involve, or facilitate any illegal, harmful, deceptive, fraudulent, or unauthorized activity. Prohibited uses
  include, without limitation, voice impersonation or cloning without explicit and lawful consent; misinformation,
  disinformation, or deception (including fake news, fraudulent calls, or presenting generated content as genuine
  recordings of real people or events); and the generation of unlawful, harmful, libelous, abusive, harassing,
  discriminatory, hateful, or privacy-invasive content. We disclaim all liability for any non-compliant use.
extra_gated_fields:
  Company or university if applicable: text
  I want to use this model for:
    type: select
    options:
      - Work
      - Studies
      - Fun
---

# Jugnu Pocket TTS (v0) — Hindi · English · Hinglish

**[▶ Listen to samples](https://altslate-labs.github.io/jugnu-pocket-tts/)** · [Code, recipe and lab notes](https://github.com/AltSlate-Labs/jugnu-pocket-tts)

A family of three small text-to-speech models that speak **Hindi, English and code-switched Hinglish** and clone a
voice from a few seconds of reference audio. Trained by [altslate labs](https://huggingface.co/altslate) with
[Kyutai's Pocket TTS / CALM recipe](https://github.com/kyutai-labs/pocket-tts).

This is a **community-trained model. It is not an official Kyutai release** and is not affiliated with Kyutai.

> **v0 is a research preview.** It gets the words right and clones adult voices well, but it was trained mostly on
> 16 kHz field recordings, has seen no children's speech and no Indian-accented English, and has not been through a
> human listening study. Read [Limitations](#limitations) before using it for anything that matters.

| Model | Folder | Layers | Use |
| --- | --- | --- | --- |
| Teacher | `teacher-24l/` | 24 | Research / distillation source. Needs classifier-free guidance 2.0, which the public `pocket-tts` runtime does not expose — sample it with the eval code in our repo. |
| Base | `base-12l/` | 12 | Distilled student, guidance baked in. Runs in the standard `pocket-tts` runtime. |
| Lite | `lite-6l/` | 6 | Distilled student, guidance baked in. The CPU / small-device model. |

## How to run

**Python API** (downloads, assembles and caches everything for you):

```bash
pip install git+https://github.com/AltSlate-Labs/jugnu-pocket-tts
huggingface-cli login   # after accepting the terms of this repo and of kyutai/pocket-tts
```

```python
from jugnu_tts import Jugnu

tts = Jugnu("lite")                           # or "base"
voice = tts.clone("my_consented_voice.wav")   # 5–15 s of clean speech
tts.speak("आपका order confirm हो गया है और delivery कल शाम तक हो जाएगी", voice, out="hello.wav")
voice.save("me.safetensors")                  # reuse later with tts.load_voice(...)
```

**Command line:**

The audio codec (Mimi) is Kyutai's and is **not redistributed here**. `assemble.py` fetches it with your own account.

1. Accept the terms of this repo and of [`kyutai/pocket-tts`](https://huggingface.co/kyutai/pocket-tts).
2. Then:

```bash
pip install pocket-tts huggingface_hub safetensors pyyaml
huggingface-cli login
huggingface-cli download altslate/jugnu-pocket-tts --include "lite-6l/*" --local-dir jugnu
python jugnu/lite-6l/assemble.py
pocket-tts generate --config jugnu/lite-6l/config.local.yaml \
    --voice my_consented_voice.wav \
    --text "कल की meeting तीन बजे है, और updated link मैंने भेज दिया है"
```

Text can be Devanagari, Latin, or mixed in one sentence. Write English words in Latin script and Hindi in
Devanagari. Romanized Hindi (`kal meeting hai`) is **not** supported. Numbers should be written out as words.

**Use a clean reference clip.** The model reproduces the acoustics of its voice prompt: with a clean studio reference
the teacher scores UTMOS 3.65 (Hindi) / 3.74 (Hinglish); with a noisy phone recording it scores ~2.0–2.6 on the same
text.

## Results

Held-out benchmark: 20 IndicVoices speakers never seen in training; the voice prompt is 5 s of a *different*
utterance by the same speaker; 150 Hindi + 150 Hinglish sentences. WER/CER by
[IndicConformer-600M](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual); the same ASR scores
**14.5% / 15.7% WER on the real recordings** of these sentences, so treat ~9–14% as the judge's floor. Speaker
similarity is WavLM-base-plus-sv cosine (English-trained, unvalidated on Hindi).

| Model | Hindi WER / CER | Hinglish WER / CER | Speaker sim | UTMOS (noisy prompts) |
| --- | --- | --- | --- | --- |
| Teacher 24L (115k steps, cfg 2.0) | 9.1% / 3.3% | 13.3% / 6.7% | 0.91 | 2.5 / 2.3 |
| Base 12L (90k distillation steps; benchmarked at 60k) | 9.0% / 3.2% | 12.1% / 5.5% | 0.91 | 2.6 / 2.4 |
| Lite 6L (112.5k distillation steps; benchmarked at 100k) | 9.0% / 3.2% | 12.2% / 5.2% | 0.91 | 2.5 / 2.4 |

Both students are still training to 200k steps; final weights will replace these. On a 16-thread CPU in the standard
runtime, Lite runs at 3.3× real time and Base at 1.5×.

Not yet measured: English WER, cross-language cloning at scale, long text / names / numbers, latency and CPU
real-time factor, any human listening test.

## Training

- **Data (647 h):** 447 h of quality-filtered Hindi from [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices)
  (2,143 speakers, 16 kHz; kept only utterances rated "excellent" with none of the dataset's 17 problem flags) and a
  200 h subset of [HiFiTTS-2](https://huggingface.co/datasets/nvidia/hifitts-2) English (44.1 kHz).
- **Hinglish** comes from IndicVoices' own annotations: transcripts mark code-switched words as `किलो [Kilo]`, and for
  half of those utterances we train on the Latin spelling. This is word-level code-switching inside Hindi sentences.
- **Alignment:** word timestamps from a Hindi wav2vec2 CTC model (Vakyansh) and Kyutai's published HiFiTTS-2 alignments.
- **Tokenizer:** one SentencePiece BPE, 4,000 tokens, Devanagari + Latin.
- **Teacher:** 24-layer FlowLM trained **from random initialisation** on frozen Mimi latents (LSD objective, lr 2e-4,
  effective batch 64, CFG dropout 0.2). We release the **115k-step** checkpoint: on this noisy corpus, training on to
  400k steps left similarity unchanged and made WER worse (9.1% → 14.3% Hindi), so longer was not better.
- **Students:** depth + guidance distillation from that teacher (Kyutai's `depth_distill` recipe; 12 and 6 layers).
- Code, configs, the full experiment log (including what went wrong) and the roadmap:
  https://github.com/AltSlate-Labs/jugnu-pocket-tts.

## Training details

| | Teacher (24L) | Base (12L) | Lite (6L) |
| --- | --- | --- | --- |
| Parameters (generator only) | 316 M | 165 M | 89 M |
| Initialisation | random | teacher head + teacher layers 0–5, 18–23 | teacher head + teacher layers 0–2, 21–23 |
| Objective | LSD flow head (1-step sampling) + end-of-speech loss 0.1 | match the teacher's backbone output (MSE), guidance 2.0 baked in | same as Base |
| Learning rate / schedule | 2e-4, cosine, 1k warm-up | 4e-4, cosine, 1k warm-up | 4e-4, cosine, 1k warm-up |
| Effective batch | 64 utterances (16 × 2 GPUs × 2 accumulation) | 64 (32 × 1 GPU × 2) | 64 (32 × 1 GPU × 2) |
| Conditioning dropout (text / voice) | 0.2 / 0.2 | none | none |
| Weight EMA | 0.999 | 0.9999 | 0.9999 |
| Steps | 400k run; **115k released** | 200k planned (90k in this release) | 200k planned (112.5k in this release) |
| Speed on RTX PRO 4500 Blackwell 32 GB | ~3.0 steps/s on 2 GPUs | ~2.5 steps/s on 1 GPU | ~3.2 steps/s on 1 GPU |
| Wall-clock | ~11 h to 115k (~37 h to 400k) | ~22 h to 200k | ~17 h to 200k |
| CPU speed (16 threads, standard runtime) | — | 1.5× real time | 3.3× real time |

**Shared:** Kyutai's Mimi codec, frozen (24 kHz, 12.5 frames/s, 32-dim continuous latents, precomputed once);
d_model 1024, 16 heads, 6-layer flow head; SentencePiece BPE with 4,000 tokens over Devanagari + Latin; AdamW
(weight decay 0.1, betas 0.9 / 0.95, gradient clip 1.0); utterances up to 30 s; the voice prompt is the part of the
same utterance before a random word boundary (max 5 s), the target is the rest.

**Data:** 451k IndicVoices Hindi utterances → 193k kept (447 h, 2,143 speakers) after requiring the dataset's
"excellent" rating, none of its 17 problem flags, no non-speech tags, 2–30 s and ≥ 3 words; 35% carry English
glosses, half of which are trained in Latin script. Plus 66k HiFiTTS-2 English utterances (200 h). 20 validation and
20 test speakers held out before anything was trained. Hindi word timestamps from a wav2vec2 CTC aligner (190,126 of
190,132 aligned); English timestamps from Kyutai's published alignments.

**Before every launch:** `training/check_codec.py` encodes and decodes held-out clips and refuses to start unless
the latents are non-degenerate and the reconstruction still transcribes (English drift 1%, Hindi 24% with this codec).

## Limitations

- **Hindi audio is band-limited.** Two-thirds of the training audio is 16 kHz field speech; Hindi sounds duller than
  English. Kyutai's codec also loses more detail on our Hindi than on English (24% vs 1% ASR drift after encode→decode).
- **No child speech, no Indian-accented English, little expressive speech** in training. Expect weak cloning of
  children, very high/low pitch, strong regional accents, shouting or whispering.
- **Hinglish is harder than Hindi** (+3–4 WER points) and is word-level only; clause-level switching is untested.
- **Occasional unintelligible generations** on hard prompts; rare words and English loanwords can be mangled.
- Metrics come from automatic judges that were themselves trained on similar data or on English.

## Responsible use

This model can imitate a voice from a short recording. Only clone voices with the speaker's **explicit, lawful
consent**. Do not use it for impersonation, fraud, or to present generated audio as a real recording. The
prohibited-use terms above (the same as Kyutai's) apply to everything in this repo. The audio samples here use a
public-domain LibriVox reader voice only.

## Acknowledgements

This work stands on other people's:

- **[Kyutai](https://kyutai.org)** — Pocket TTS, the Mimi codec, the CALM recipe and the open training code
  ([repo](https://github.com/kyutai-labs/pocket-tts), [paper](https://arxiv.org/abs/2509.06926)). Codec weights CC BY 4.0.
- **[AI4Bharat](https://ai4bharat.iitm.ac.in)** — IndicVoices (CC BY 4.0) and IndicConformer (MIT).
- **NVIDIA** and **LibriVox** volunteers — HiFiTTS-2 (CC BY 4.0); Kyutai for the published alignments.
- **Vakyansh / Harveen Chadha** — the Hindi wav2vec2 model used for forced alignment.
- **Singh, Singh & Kadyan (2025)** — HiACC, used only to test child-voice cloning (CC BY 4.0).
- **OpenAI Whisper**, **Microsoft WavLM**, **UTMOS** — evaluation.

## License

Our weights, tokenizer and configs: **CC BY 4.0**. Code: MIT (on GitHub). Kyutai's codec is fetched from their repo under their terms.
Please credit "Jugnu Pocket TTS, altslate labs" and the projects above.
