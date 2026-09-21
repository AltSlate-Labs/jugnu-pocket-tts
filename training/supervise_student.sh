#!/bin/bash
# usage: supervise_student.sh <name> <gpu>   e.g. lite6 3
# Runs a distillation to 200k steps, auto-resuming from the latest checkpoint after a crash (max 8 restarts).
name=$1; gpu=$2; T=~/hindi-tts
cd $T/pocket-tts; export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for attempt in $(seq 1 8); do
  [ -s $T/runs/${name}_v0/checkpoint_00200000.pt ] && break
  echo "=== attempt $attempt $(date)" >> $T/train_${name}_v0.log
  CUDA_VISIBLE_DEVICES=$gpu uv run python training/train.py $T/configs/distill_${name}_v0.yaml >> $T/train_${name}_v0.log 2>&1
  sleep 30
done
echo "SUPERVISOR_EXIT $name $(date)" >> $T/chain_status.txt
