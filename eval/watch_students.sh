#!/bin/bash
# Benchmark the distilled students every 50k steps. Guidance is baked in, so they are sampled at cfg 1.0.
T=~/hindi-tts
declare -A GPU=([base12]=3 [lite6]=3)  # Base's trainer keeps ~31 GB reserved on GPU 2, evals cannot share it
while true; do
  for arm in base12 lite6; do
    for ck in $(ls $T/runs/${arm}_v0/checkpoint_*.pt 2>/dev/null); do
      step=$(basename $ck .pt | sed 's/checkpoint_0*//')
      [ $((step % 50000)) -eq 0 ] || continue
      out=$T/runs/${arm}_v0/eval_hi/step$(printf %08d $step)_cfg1.0
      [ -e $out/results_indicconf.json ] && continue
      bash $T/run_eval.sh $arm ${GPU[$arm]} --checkpoint $ck --use-ema --cfg 1.0 --batch-size 16
      OMP_NUM_THREADS=32 ~/hindi-stt/nemoenv/bin/python $T/score_indicconf.py $out >> $T/eval_${arm}.log 2>&1
    done
  done
  sleep 180
done
