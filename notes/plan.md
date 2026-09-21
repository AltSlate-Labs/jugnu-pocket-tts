# Hindi–English–Hinglish TTS: Pocket/CALM Teacher–Student Project Plan

**Date:** 18 September 2026  
**Organization:** altslate labs  
**Compute:** AWS `g7.24xlarge`  
**Status:** Proposed implementation plan; no training runs have been executed for this document.

## 1. Project objective

Build a family of three text-to-speech models that support Hindi, English and natural Hinglish, with voice cloning from a short reference recording. Train a capable teacher, then distill it into two smaller models for different deployment budgets.

The intended product capabilities are:

- Natural speech in Hindi and English, including Indian English.
- Code switching within a sentence, with appropriate pronunciation and rhythm.
- Cloning of speakers who were not included in training.
- Cross-language cloning while retaining speaker identity.
- Streaming audio generation and a compact option for CPU deployment.

The selected direction is a **Pocket TTS/CALM-style continuous-audio language model**. This document proposes an adaptation for the target languages; it does not claim that the upstream model already establishes Hindi or Hinglish quality.

## 2. The three-model family

| Model | Proposed depth | Purpose | Evidence status |
| --- | --- | --- | --- |
| Full | 24-layer teacher | Highest-quality candidate and source for distillation | Matches the teacher depth in the current Pocket training recipe |
| Base | 12-layer student | Balance between quality, latency and memory | Our proposed additional experiment |
| Lite | 6-layer student | CPU and constrained-device deployment | Matches the student depth in the current Pocket training recipe |

The upstream recipe trains a 24-layer teacher and distills it into a 6-layer student, including classifier-free guidance in the distillation process. The 12-layer model is an extension to evaluate. [Pocket training documentation](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md)

All three models should share:

- The text tokenizer and text-normalization conventions.
- A frozen audio autoencoder and its latent representation.
- The reference-audio interface used for voice cloning.
- The evaluation suite and output-audio format.

Measure exact parameter counts after choosing dimensions and implementing the configurations. Report generator parameters, audio-autoencoder parameters and total deployed parameters separately. A layer count alone does not determine model size or speed.

The default proposal is to distill Base and Lite independently from Full. A serial Full → Base → Lite experiment can be considered later if there is evidence that it improves the result.

## 3. Architecture and what “from scratch” means

The generator models speech as a sequence of continuous audio representations. A causal Transformer uses text, reference audio and previously generated audio to provide context for a small generative output head. An audio decoder turns the generated representations into a waveform. The CALM paper motivates continuous audio modeling and efficient generative heads. [CALM paper](https://arxiv.org/abs/2509.06926)

| Component | Function | Project treatment |
| --- | --- | --- |
| Text normalization and tokenizer | Represent Hindi, English, punctuation and mixed text | Build conventions and tokenizer for the combined corpus |
| Audio encoder | Convert training/reference recordings into continuous latents | Train within this project |
| Audio decoder | Reconstruct waveforms from latents | Train within this project; share across the family |
| Teacher Transformer and generative head | Learn reference-conditioned speech generation | Train from random initialization |
| Student generators | Approximate the teacher with lower inference cost | Distill from our own teacher |

**Scope:** all deployed synthesis weights are to be trained within this project. Students may inherit weights or supervision from our own trained teacher.

Pocket’s published generator recipe works with Mimi audio representations. It does not, by itself, establish a complete from-scratch audio-autoencoder training recipe. Auditing and implementing that part is a separate work item. [Pocket training documentation](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md)

Keep a dependency record for pretrained tools used in transcription, alignment, filtering or evaluation. If a pretrained model provides a training loss or semantic supervision, disclose that separately; distinguish random initialization of deployed weights from absence of pretrained supervision.

### Audio-autoencoder gate

Before committing to generator training, reconstruct held-out recordings through the encoder and decoder. Evaluate:

- Hindi phonetic distinctions and English consonants.
- Naturalness at Hindi–English switching points.
- Speaker identity and accent preservation.
- Pauses, breaths and speaking rhythm.
- Reconstruction artifacts and decoder runtime.

Compare original and reconstructed recordings in blind listening tests. The autoencoder must preserve the information the generator will need to learn.

Freeze the autoencoder checkpoint, latent dimensions, frame rate and normalization statistics before generating a large latent cache. Changing the representation requires regenerating that cache and checking generator compatibility.

## 4. Data plan

### Initial collection targets

These are proposed budgets for cleaned, usable training data, not published sufficiency thresholds or quantities already available.

| Stage | Proposed data | Purpose |
| --- | --- | --- |
| Pilot | About 250 hours: 100 Hindi, 100 English, 50 natural Hinglish | Validate the representation, data pipeline and teacher learning |
| Expanded teacher | Approximately 2,000–5,000 hours total, with substantial coverage in each category | Improve robustness, cloning and expression according to pilot results |
| Targeted additions | Data selected from measured failure categories | Address gaps such as names, switching points or particular accents |

Count unique speech hours, speakers and sessions separately. Repetition through oversampling does not increase unique data. Prioritize speaker diversity alongside sufficient material per speaker to form reference/target examples.

### Starting sources

| Source | Verified scope | Intended contribution |
| --- | --- | --- |
| [LibriTTS-R](https://www.openslr.org/141/) | Approximately 585 hours across the complete English corpus | English speech with transcripts and speaker identities; preserve development/test splits |
| [IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r) | 1,704 hours across 22 Indian languages | Indian speaker diversity; choose subsets deliberately |
| [IndicVoices-R Hindi](https://arxiv.org/html/2409.05356v2) | 74.6 hours from 399 speakers in the published statistics | Initial Hindi diversity; additional Hindi collection is needed for the proposed pilot target |
| Own or commissioned recordings | Targeted collection, subject to acquisition | Natural Hinglish, Indian English, bilingual speakers and product-specific text |

Published corpus totals include material reserved for evaluation and material that may fail our filters. Establish usable training totals after processing. LibriTTS-R and IndicVoices-R list CC BY 4.0 licenses; retain dataset versions and provenance alongside the manifests.

### Bilingual recording design

Collect Hindi, English and Hinglish from the same speakers where possible. This provides evidence of how a voice behaves across languages and reduces confounding between language and identity.

Include:

- Natural conversation and explanatory speech, alongside read passages.
- Sentences that switch language within a clause or phrase.
- Names, addresses, dates, amounts, abbreviations and technical vocabulary.
- Questions, statements, emphasis, varied pacing and different sentence lengths.
- Multiple recording sessions per speaker where practical.

Example Hinglish prompt:

> कल की meeting तीन बजे reschedule कर देना, और updated link मुझे भेज देना।

Use fluent bilingual speakers to review scripts and recordings. Preserve the spoken wording in transcripts instead of translating or rewriting it into formal Hindi.

## 5. Text and audio preparation

### Text representation

Train one tokenizer over the combined corpus, covering Devanagari, Latin characters, punctuation and the required numeric forms. Keep both verbatim transcripts and any normalized training text, with traceable transformations between them.

Handle these input cases explicitly:

| Input type | Example | Design requirement |
| --- | --- | --- |
| Hindi | कल बैठक है। | Preserve Hindi pronunciation and script handling |
| English | The meeting is tomorrow. | Cover the intended English accents and vocabulary |
| Mixed script | कल meeting है। | Model natural spoken code switching |
| Romanized Hindi | Kal meeting hai. | Evaluate a dedicated normalization/transliteration approach |

Romanized Hindi is ambiguous and should have its own test set. Do not silently transliterate every Latin word: many are English words that should remain English.

### Audio preparation

- Preserve original audio and create versioned training copies.
- Select the training sample rate to match the chosen autoencoder configuration.
- Initially favor short, coherent segments containing a single speaker; roughly 3–15 seconds is a working preparation target.
- Filter severe clipping, overlapping speech, music, bad cuts and transcript mismatches.
- Inspect restoration or denoising outputs for changes to voice identity and pronunciation.
- Deduplicate across datasets before creating development and test splits.

The current Pocket recipe requires word timestamps for matching cropped audio to text. Build and audit multilingual alignment, with extra checks around code-switch boundaries. [Pocket data preparation](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md)

### Manifest fields

| Field | Purpose |
| --- | --- |
| `utterance_id`, `audio_path` | Stable sample identity and audio location |
| `speaker_id`, `session_id` | Speaker pairing, split integrity and session tracking |
| `text_verbatim`, `text_normalized` | Spoken wording and explicit preprocessing output |
| `speech_category` | Hindi, English or Hinglish, with a documented labeling rule |
| `language_spans` | Optional word/span language labels for mixed speech |
| `word_timestamps` | Alignments used by cropping and validation |
| `duration_s`, `sample_rate` | Audio properties |
| `source`, `source_version`, `license_or_permission` | Provenance |
| `split`, `quality_flags` | Training/evaluation assignment and filtering decisions |

Use stable anonymized speaker IDs. Record any uncertainty in speaker grouping so that it can be checked before evaluating zero-shot cloning.

## 6. Training sequence

### Stage A — Audit and freeze the implementation

Pin an upstream commit and inspect the actual model, generator training, distillation and audio-representation code. Confirm which weights are loaded automatically. Create a component inventory and make random initialization explicit for the teacher and audio-autoencoder workstreams.

Retain the upstream generative-head objective initially. Depth distillation, guidance distillation and reducing the number of generation steps are related but distinct choices; introduce changes one at a time.

**Deliverable:** versioned architecture configuration, dependency inventory and reproducible environment.

### Stage B — Prepare the pilot and train the audio autoencoder

Build the pilot manifest and separate development and test speakers before training. Train the audio autoencoder on the permitted audio pool, then pass the reconstruction gate in Section 3. Freeze it before caching training latents.

**Deliverable:** audio-autoencoder checkpoint, reconstruction report and versioned latent cache.

### Stage C — Train the 24-layer teacher

Train with reference-audio conditioning from the start. Construct training examples using the same speaker for reference and target; avoid making the target waveform available as the reference for the very segment being predicted.

Track loss, intelligibility, voice similarity and listening samples separately for Hindi, English and Hinglish. Inspect stopping errors, repeated words, missing words and voice changes.

Use a sampler that controls both language exposure and domination by speakers with unusually large amounts of data. A 40% Hindi / 40% English / 20% Hinglish sampling mix is a possible starting hypothesis, to be adjusted on development results and actual corpus diversity.

**Deliverable:** a teacher that produces intelligible speech and measurable cloning on unfamiliar speakers.

### Stage D — Scale the teacher

Expand the corpus according to observed failures. Preserve the evaluation splits, deduplication rules and preprocessing versions. Compare improvements on each speech category before changing architecture or increasing model size.

**Deliverable:** a teacher with stable quality across all target categories and a documented operating point for inference.

### Stage E — Distill Base and Lite

Distill each student from the selected teacher, preserving the frozen audio representation and reference interface. Maintain coverage of all three speech categories during distillation.

Evaluate each student against the teacher using identical prompts, references and deployment conditions. Measure the complete synthesis pipeline, including waveform decoding.

**Deliverable:** two student candidates, each with a quality/latency/memory comparison against Full.

## 7. Compute plan for g7.24xlarge

AWS lists four RTX PRO 4500 Blackwell GPUs with 32 GB per GPU, 96 vCPUs and 384 GiB system memory for this instance. [AWS G7 specifications](https://aws.amazon.com/ec2/instance-types/g7/)

Proposed starting approach:

- Use all four GPUs for distributed training of the active model.
- Use BF16 where supported and numerically stable; retain higher precision where required.
- Select microbatch size by measured per-GPU memory, then use gradient accumulation to reach the desired effective batch.
- Use activation checkpointing when its memory saving is needed.
- Keep data decoding and storage throughput visible in profiling.
- Store durable checkpoints and data manifests outside ephemeral instance storage.

Run an initial throughput profile after warm-up. Record peak GPU memory, updates per second, audio seconds processed per second and time spent waiting for input data. Estimate training duration from these measurements plus evaluation and checkpoint overhead.

Do not transfer H100 timing numbers directly to this instance. The 128 GB aggregate GPU memory is distributed across four devices; the selected parallelization method determines how it can be used.

## 8. Evaluation and promotion gates

### Evaluation split

Create a development set for tuning and a separate locked test set. For the initial test panel, aim for about 60 held-out speakers spanning Hindi-dominant, English-dominant and bilingual speakers. This is a proposed panel size, subject to data availability.

Keep every test speaker out of audio-autoencoder, generator and distillation training where the claim is fully unseen-speaker generalization. Also check speaker and recording overlap across source corpora.

Use separate utterances for the reference clip and target evaluation. Test at least two reference lengths, such as 5 and 10 seconds, and record inference randomness for repeatable comparisons.

| Test | What it reveals |
| --- | --- |
| Hindi reference → Hindi output | Hindi pronunciation and same-language cloning |
| English reference → English output | English pronunciation and same-language cloning |
| Hindi reference → English output | Cross-language voice preservation |
| English reference → Hindi output | Cross-language pronunciation and voice preservation |
| Hindi/English reference → Hinglish output | Voice stability and pronunciation during switching |
| Long text, names, numbers and mixed-script inputs | Reliability beyond ordinary short sentences |

### Measures

| Dimension | Measurement |
| --- | --- |
| Intelligibility | Language-appropriate WER/CER plus manual review of sampled errors |
| Code switching | Word accuracy near switches, pronunciation review and skipped-word rate |
| Cloning | Speaker-similarity scores plus blind human judgments |
| Naturalness | Native/bilingual listener ratings and paired preferences |
| Reliability | Silence, truncation, repetition, missing words and speaker drift |
| Deployment | Time to first audio, real-time factor, peak RAM/VRAM and sustained throughput |

Report results separately for Hindi, English and Hinglish. Run the transcription evaluator on original recordings as a reference for ASR error. Treat learned speech-quality and speaker metrics as supporting evidence; validate their behavior on the target languages.

### Promotion rules

- **Autoencoder:** reconstructed speech preserves pronunciation and speaker identity sufficiently for teacher training.
- **Teacher:** meets the quality targets on each category, including unseen speakers and mixed sentences.
- **Students:** demonstrate a useful measured serving improvement with acceptable quality retention in every category.
- **Release:** passes the locked evaluation and documents known failures, data sources and supported inputs.

Choose numerical acceptance margins on development data before testing final candidates. A reduction in layer count does not itself establish a useful product improvement.

## 9. Proposed implementation deliverables

The following are planned repository areas, not files implemented by this document.

| Area | Deliverable |
| --- | --- |
| `data/` | Source inventory, manifest schema, split definitions and dataset versions |
| `preprocessing/` | Audio checks, text normalization, alignment and deduplication |
| `configs/` | Audio-autoencoder, teacher, Base and Lite configurations |
| `training/` | Reproducible initialization, training and resumption entry points |
| `distillation/` | Teacher loading, student initialization and distillation runs |
| `evaluation/` | Language-stratified quality, cloning and reliability evaluation |
| `serving/` | Streaming interface and CPU/GPU benchmarking |
| `reports/` | Experiment records, comparisons and model cards |

Immediate work order:

1. Pin and audit the Pocket/CALM implementation and pretrained dependencies.
2. Resolve the from-scratch audio-autoencoder recipe and its interface.
3. Inventory available Hindi, English and Hinglish recordings.
4. Define speaker-disjoint development and test sets.
5. Build the pilot manifest, normalization and alignment checks.
6. Train and evaluate the audio autoencoder.
7. Profile and train the teacher on the pilot.
8. Expand the teacher only after diagnosing pilot failures.
9. Distill Base and Lite from the selected teacher.
10. Compare all three models on quality, cloning and actual deployment cost.

## 10. Sources and claim boundaries

- [Pocket TTS repository](https://github.com/kyutai-labs/pocket-tts) — implementation and release context.
- [Pocket training documentation](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md) — teacher/student recipe, preparation requirements and upstream training workflow.
- [Continuous Audio Language Models](https://arxiv.org/abs/2509.06926) — continuous-audio modeling and generative-head foundation.
- [AWS G7 specifications](https://aws.amazon.com/ec2/instance-types/g7/) — hardware specifications.
- [LibriTTS-R](https://www.openslr.org/141/) — English corpus description and license.
- [IndicVoices-R dataset card](https://huggingface.co/datasets/ai4bharat/indicvoices_r) — dataset access, description and license.
- [IndicVoices-R paper](https://arxiv.org/html/2409.05356v2) — language-specific hours, speakers and evaluation design.

The 12-layer model, pilot composition, sampling mix, evaluation-panel size and expanded-data budget are project proposals. Upstream English results do not establish Hindi/Hinglish performance or training time on g7.24xlarge. Recheck the pinned implementation before turning these design decisions into launch configurations.

## 11. Review addendum (2026-09-19)

Review notes added when the plan was filed in this repo. Upstream claims about the Pocket training README were not re-verified in this review. Compute status at filing: all four GPUs on the box are free.

1. **Audio autoencoder is the highest-risk item and comes first in the training order.** Pocket's recipe assumes Mimi, which was trained on far more audio than this plan has and used semantic distillation from a pretrained speech model. Proposed control: run the pilot teacher on frozen Mimi latents as a diagnostic arm, so a bad pilot result can be attributed to either generator/data or our codec. The codec needs no transcripts, so its training pool can be much larger than the 250 h pilot (e.g. the 819 h IndicVoices Hindi already on the box). If "all deployed weights trained within this project" is a hard requirement, the Mimi arm stays diagnostic only.
2. **Natural Hinglish is the real data bottleneck.** No listed public source supplies it; the 50 h pilot share is entirely own/commissioned recording, the longest-lead item. Start it in parallel with the Stage A audit rather than third in the work order.
3. **Word timestamps are their own work item.** Code-switch boundaries are where aligners fail. The IndicConformer CTC head ported for the Hindi STT project (`notes/hindi-stt/`) is a candidate aligner for Devanagari; mixed-script lines likely need per-span routing to separate Hindi/English aligners or transliteration before alignment.
4. **Evaluation ASR is weak where it matters.** In our STT baselines Whisper-large-v3 scored 44.5 WER on spontaneous Hindi, and IndicConformer has in-distribution caveats on IndicVoices. Word accuracy near switches may be mostly ASR noise; budget human review as the primary Hinglish metric.
5. **IndicVoices-R Hindi likely overlaps the IndicVoices download already on the box** (flagged as a re-release in the STT notes). Deduplicate by speaker across the two before building the 60-speaker test panel.
