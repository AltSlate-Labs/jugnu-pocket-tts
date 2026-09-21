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
| Base 12L | _pending_ | _pending_ | _pending_ | _pending_ |
| Lite 6L | _pending_ | _pending_ | _pending_ | _pending_ |

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

Our weights, tokenizer and configs: **CC BY 4.0**. Kyutai's codec is fetched from their repo under their terms.
Please credit "Jugnu Pocket TTS, altslate labs" and the projects above.
