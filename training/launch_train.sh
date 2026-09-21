#!/bin/bash
# Finalize manifests, train the shared tokenizer, write the model config, launch both teacher arms.
set -e
# Codec reconstruction gate (decision D14) runs below, after the model config is written.
cd ~/hindi-tts/pocket-tts
export PATH=$HOME/.local/bin:$PATH
T=~/hindi-tts
uv run python $T/build_manifests.py finalize $T/data/iv_hi data/hifitts2_200h $T/data/v0
uv run python -m training.scripts.train_tokenizer $T/data/v0/tokenizer $T/data/v0/train_aligned.jsonl
# Model config: released 24L architecture + gated kyutai/pocket-tts weights (working Mimi encoder; the ungated file zeroes it), our tokenizer.
sed -e "s#^weights_path:.*#weights_path: hf://kyutai/pocket-tts/languages/english_2026-04_24l/model.safetensors@492522650173a0653b7575cdc25ae09810e5d741#" \
    -e "s#tokenizer_path:.*#tokenizer_path: $T/data/v0/tokenizer.model#" \
    pocket_tts/config/english_2026-04_24l.yaml > $T/configs/hinglish_24l.yaml
CUDA_VISIBLE_DEVICES=0 uv run python $T/check_codec.py $T/configs/scratch_v0.yaml $T/data/v0/valid_aligned.jsonl
export OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=0,1 nohup setsid uv run torchrun --nproc-per-node 2 --master-port 29501 training/train.py $T/configs/scratch_v0.yaml > $T/train_scratch_v0.log 2>&1 </dev/null &
CUDA_VISIBLE_DEVICES=2,3 nohup setsid uv run torchrun --nproc-per-node 2 --master-port 29502 training/train.py $T/configs/ftlang_v0.yaml > $T/train_ftlang_v0.log 2>&1 </dev/null &
echo LAUNCHED
