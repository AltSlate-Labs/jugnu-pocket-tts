#!/bin/bash
# After the scratch teacher finishes (checkpoint_00400000.pt), distill Base (GPU 2) and Lite (GPU 3).
T=~/hindi-tts
until [ -s $T/runs/scratch_v0/checkpoint_00400000.pt ] && ! pgrep -f "[t]raining/train.py" >/dev/null; do sleep 120; done
sleep 120
mkdir -p $T/final_v0/scratch_400k && cp $T/runs/scratch_v0/checkpoint_00400000.pt $T/runs/scratch_v0/model.safetensors $T/final_v0/scratch_400k/
cd $T/pocket-tts; export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=2 nohup setsid uv run python training/train.py $T/configs/distill_base12_v0.yaml > $T/train_base12_v0.log 2>&1 </dev/null &
CUDA_VISIBLE_DEVICES=3 nohup setsid uv run python training/train.py $T/configs/distill_lite6_v0.yaml > $T/train_lite6_v0.log 2>&1 </dev/null &
echo "DISTILL_LAUNCHED $(date)" > $T/chain_status.txt
pkill -f "[w]atch_eval.sh"
nohup setsid bash $T/watch_students.sh > $T/watch_students.log 2>&1 </dev/null &
