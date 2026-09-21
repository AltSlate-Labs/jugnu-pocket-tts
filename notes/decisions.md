# Hindi–English–Hinglish TTS — decisions log

Newest at the bottom. Each entry: decision, why, what it costs / what would reverse it.
Plan: [plan.md](plan.md). Runs: [experiments.md](experiments.md).

## 2026-09-19

**D1. Pin upstream `kyutai-labs/pocket-tts` at `d9206671923c50159a00ff07660e9a759b3b68d3`** (2026-09-18).
Audit result: the repo ships a complete training stack (`training/`): 24L teacher from scratch (`scratch.yaml`),
6L depth+CFG distillation (`depth_distill.yaml`), CTC forced alignment, tokenizer training, latent precompute,
WER / speaker-sim / UTMOS eval. It does **not** ship codec training: every config loads the released, frozen
Mimi weights. Confirms plan §3's caveat.

**D2. v0 uses the released frozen Mimi codec.** Why: no upstream codec recipe exists, and a frozen-Mimi run is the
diagnostic arm the review addendum asked for — it tells us whether the generator + our data work before we risk a
codec of our own. Cost: v0 is *not* "all deployed weights trained in-project" (generator is from random init; codec is
pretrained by Kyutai, CC-BY 4.0). From-scratch codec stays a separate, later workstream.

**D3. v0 Hindi data = IndicVoices Hindi already on the box (16 kHz), quality-filtered.** Why: IndicVoices-R (the
48 kHz restored TTS release) and Rasa are HF-gated and the account is not on the authorized list (403); IndicVoices
is on disk and licensed. Filter: `decision == excellent`, none of 17 verification flags (noise, chatter, echo, low
volume, mispronunciation, skipped words, …), no non-speech tags in the transcript, 2–30 s, ≥3 words.
Cost: Hindi speech is band-limited to 8 kHz audio bandwidth, so Hindi output will sound duller than English.
Reverse when: IndicVoices-R access is granted (user action: accept terms at
https://huggingface.co/datasets/ai4bharat/indicvoices_r and https://huggingface.co/datasets/ai4bharat/Rasa).

**D4. v0 English data = HiFiTTS-2 200 h subset** via upstream `prepare_data.py` with Kyutai's published word
alignments. Why: zero-friction, known-good with this recipe. Cost: US/UK audiobook English, no Indian English yet.

**D5. Hinglish text comes from IndicVoices' own `[English]` glosses.** `unsanitized_normalized` marks code-switched
words as `किलो [Kilo]`. We align on the Devanagari transcript, then for utterances with glosses swap in the Latin
spelling for a deterministic ~50% of them, so the model sees both `किलो` and `Kilo` for the same audio. Only
single-word Latin glosses that line up 1:1 with the clean transcript are used. This is word-level code-switching
inside Hindi sentences — not the clause-level natural Hinglish the plan wants; that still needs recording.

**D6. Hindi forced aligner = `Harveenchadha/vakyansh-wav2vec2-hindi-him-4200`** (char-level Wav2Vec2 CTC, ungated,
drops into upstream `align_data.py`). `ai4bharat/indicwav2vec-hindi` is gated. Pretrained tool used for data prep
only — listed in the dependency record.

**D7. One shared SentencePiece BPE tokenizer, vocab 4000**, trained on the combined Hindi + mixed + English
transcripts (upstream default size, so `n_bins` is unchanged).

**D8. Speaker-disjoint holdout from the start.** 20 valid + 20 test IndicVoices speakers (gender-balanced) are removed
from training. Preset voices and cloning tests use test speakers only.

**D9. Three preset voices (male, female, kid) = three reference clips**, not three models. Male/female come from
held-out test speakers. Kid is a known gap: IndicVoices speakers are all 18+, so no child speech is in training;
v0 kid voice will be tested with a child reference clip and reported honestly.

**D11. Run two teacher arms in parallel, 2 GPUs each** (batch 16 × accum 2 × 2 GPUs = effective 64 for both).
Arm A `scratch`: FlowLM from random init — the plan-compliant model. Arm B `ftlang`: upstream
`finetune_language.yaml` — starts from Kyutai's released English 24L teacher with a fresh text embedding; upstream
reports same WER ~2.5× sooner on Czech. Why: upstream timings show 4 GPUs is only ~20% faster than 2, so splitting
costs little and gives a direct benchmark of "from scratch" vs "inherit English". Arm B's weights are not
in-project-only; it is a benchmark and a fallback, not a replacement for Arm A.

**D12. Kid preset voice = HiACC speaker CH26 (girl, 11), clips CH26045 + CH26044 concatenated (~14 s, 16 kHz).**
HiACC = Hinglish Adult & Children Code-switched corpus (Singh, Singh & Kadyan 2025, CC BY 4.0, Zenodo
10.5281/zenodo.15551669): 20 native-Hindi children aged 10–14, 2.04 h. Pulled from the ungated HF mirror
`Trelis/hiacc-child-test-eval` (test split, 372 clips). Clip chosen by SNR/pitch, **not yet listened to**. The model
has seen no child speech, so this tests out-of-distribution cloning. Later: add HiACC children train/val (keep this
test split held out) plus Samrómur Children (131 h, CC BY 4.0, Icelandic) for child acoustics. Male/female presets =
held-out IndicVoices test speakers S4258262800387709 / S4257983600364803 (18–30, read speech).

**D10. Training schedule: upstream `scratch.yaml` unchanged (lr 2e-4, effective batch 64, cosine to 400k)**, batch
16 × 4 GPUs. Why: upstream notes say compressing the schedule stalls the acoustic-quality transition. We evaluate
intermediate checkpoints (WER expected to drop ~15–50k steps) rather than shortening the run.

**D13. (2026-09-19) D2's weight source is void: the ungated Kyutai file has a zeroed Mimi encoder.** A working
frozen-Mimi run needs the gated `kyutai/pocket-tts` weights (user must accept the terms on their HF account — a
deliberate access control, not worked around). Until then the options are: (a) user accepts → relaunch both arms
unchanged, first speech benchmark ~4 h later; (b) train our own encoder/codec in-project, which the plan wants
anyway but is days, not hours. Recommendation: (a) now for v0, (b) as the planned codec workstream.

**D14. No training launch without the codec reconstruction gate.** `check_codec.py` encodes→decodes held-out
Hindi, English and Hinglish clips and fails unless latents are non-degenerate and the reconstruction's ASR
transcript matches the original's.

**D15. (2026-09-19) User accepted the `kyutai/pocket-tts` terms → v0 proceeds on the gated frozen Mimi (D13 option a).**
Codec gate limits: English drift < 10%, Hindi drift < 35% (whisper-turbo is a weak Hindi judge, so the Hindi bound
only catches a broken codec; the measured 23.8% is logged as a finding, see experiments E3). Own-codec work stays the
next workstream, now with a measured motivation.

**D16. (2026-09-19) Own-codec work is deferred until v0 training completes** (user decision). It is tracked as
phases C0–C4 in [roadmap.md](roadmap.md): scoping → fine-tune control → from-scratch codec → reconstruction +
modellability gates → freeze and retrain the family. No scoping subagent launched now.

**D17. (2026-09-19) Both arms stopped at step 115k at the user's request (GPUs needed elsewhere).** Resumable state
archived in `~/hindi-tts/final_v0/`. Distillation is deferred: students are better distilled from a teacher that has
passed the 150–200k quality transition. Open question for the user: is v0 = the 115k teachers, or resume later.

**D18. (2026-09-19 15:41 UTC) Arm A (`scratch_v0`) resumed on GPUs 2,3 from step 115,192** (user freed two GPUs;
GPU 0 runs the user's jugnu job, GPU 1 left alone). Arm A chosen over Arm B: it is the plan-compliant generator,
matched Arm B on Hindi WER and speaker similarity at 115k, and its UTMOS was still rising. Same config, LR schedule
continues (cosine to 400k). Arm B stays parked at 115,511 (resumable). Eval watcher restarted on GPU 2.

**D19. (2026-09-19) Distillation is chained to the end of the teacher run, no further confirmation (user instruction).**
Teacher = Arm A run to its full 400k steps (fully decayed LR → best distillation source). Then, automatically
(`chain_distill.sh`): Base 12L on GPU 2 and Lite 6L on GPU 3, one GPU each, batch 64, upstream `depth_distill.yaml`
settings (cfg 2.0 baked in, lr 4e-4 cosine, 200k steps, no conditioning dropout, EMA 0.9999). Students benchmarked
every 50k steps at cfg 1.0 (`watch_students.sh`). Only GPUs 2,3 are used. Smoke-tested for 40 steps each against the
115k teacher: 6L seeded from teacher layers [0-2, 21-23], 12L from [0-5, 18-23]; both train and checkpoint.

**D20. (2026-09-20) Naturalness is judged with clean reference voices from now on.** E6 showed UTMOS ~2.6 on the
standard benchmark is capped by the noisy IndicVoices prompts (real recordings score 2.0); with a clean studio
reference the 200k scratch teacher reaches 3.65 (Hindi) / 3.74 (Hinglish). The standard benchmark stays as the
intelligibility + cloning test; a clean-reference track is added for quality. Preset voices must be replaced with
clean (ideally consented, studio) recordings before any public samples. Teacher continues to 400k as chained (D19).

**D21. (2026-09-21) Students are distilled from the 115k scratch teacher, not the 400k one.** The 400k teacher
benchmarked at 14.3% / 18.9% IC WER (Hindi / Hinglish) against 9.1% / 13.3% at 115k, with identical speaker
similarity and prompt-capped UTMOS; no sampling setting recovers the gap (E5 sweeps). The user's instruction was to
distill after the run without asking; choosing the better archived teacher serves that instruction's intent and was
announced in advance. Cost: the 115k checkpoint was taken mid-schedule (lr 1.6e-4), mitigated by distilling from its
EMA weights (`distill_teacher_use_ema` default). Distillation batch is 32 × 2 accumulation on one GPU each (D19's
batch 64 OOMs in validation).

**D22. (2026-09-21) Hugging Face release design (staged, NOT pushed — awaiting user go-ahead).** Our export bundles
Kyutai's Mimi codec (87 tensors incl. the voice-cloning encoder) with our FlowLM (271 tensors, 316 M params).
Kyutai's weights are CC BY 4.0 but gated behind a prohibited-use acknowledgement (no cloning without explicit lawful
consent, no deception/fraud, …). We therefore publish **only our FlowLM weights** + tokenizer + config + `assemble.py`,
which downloads the codec from `kyutai/pocket-tts` with the user's own accepted access and writes a normal bundle
locally (tested end-to-end on CPU through the public `pocket_tts.TTSModel` API: 4.6 s of audio in 6.1 s, 16 threads).
Our repo should carry the same prohibited-use gate. Acknowledge: Kyutai (Pocket TTS, Mimi, CALM recipe), AI4Bharat
(IndicVoices, IndicConformer), NVIDIA + LibriVox (HiFiTTS-2), HiACC authors, Vakyansh aligner, OpenAI Whisper,
Microsoft WavLM, UTMOS. Caveat found while testing: the public inference API has no classifier-free guidance, and the
teacher needs cfg 2.0 (at cfg 1.0 it scores 21.8% / 25.0% WER) — so the teacher is a research / distillation artefact;
the models meant for end users are the distilled students, which bake guidance in.

**D23. (2026-09-21) Release decisions (user):** publish **all three models together** as one gated Hugging Face
repo **`altslate/jugnu-pocket-tts`** (subfolders `teacher-24l/`, `base-12l/`, `lite-6l/`), with Kyutai's
prohibited-use acknowledgement as the gate, **clean-voice samples only** (public-domain LibriVox reader; no
IndicVoices speakers, no child voice), and acknowledgements to every upstream project. "vaani-pocket-tts" was
rejected because Vaani is the ARTPARK/IISc dataset's name and we used none of its data. Push happens after the
students' benchmarks are in; model card draft at `hindi-tts/release/README.md`. Our weights: CC BY 4.0.

**D24. (2026-09-21) GitHub repo = `AltSlate-Labs/jugnu-pocket-tts`** (user), same name as the HF repo. Public, MIT
for our code + NOTICE with CC BY attributions. Created only when the release is pushed. Open: whether commits may
carry Claude attribution (the public `AltSlate-Labs/jugnu` repo forbids it) — default to **no attribution** unless
the user says otherwise.

**D25. (2026-09-21) The GitHub repo gets a GitHub Pages site** (user): `https://altslate-labs.github.io/jugnu-pocket-tts/`.
Plan: a single static page served from `docs/` — what it is, audio player grid (clean-voice samples only: Hindi /
Hinglish / English × Teacher / Base / Lite), benchmark table, how to run, limitations, responsible use,
acknowledgements, links to the HF repo and the experiment log. Samples committed as small mono files; no dataset
audio, no real-speaker clones.
