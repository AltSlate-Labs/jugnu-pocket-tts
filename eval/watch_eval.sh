#!/bin/bash
# Evaluate every 25k-step checkpoint of both arms as it appears (EMA weights, 150 items per set),
# then re-score with IndicConformer. One eval at a time; each arm is generated on its own GPU.
T=~/hindi-tts
declare -A GPU=([scratch]=0 [ftlang]=2)
while true; do
  for arm in scratch ftlang; do
    for ck in $(ls $T/runs/${arm}_v0/checkpoint_*.pt 2>/dev/null); do
      step=$(basename $ck .pt | sed 's/checkpoint_0*//')
      [ $((step % 25000)) -eq 0 ] || continue
      out=$T/runs/${arm}_v0/eval_hi/step$(printf %08d $step)_cfg2.0
      [ -e $out/results_indicconf.json ] && continue
      bash $T/run_eval.sh $arm ${GPU[$arm]} --checkpoint $ck --use-ema --batch-size 8
      OMP_NUM_THREADS=32 ~/hindi-stt/nemoenv/bin/python $T/score_indicconf.py $out >> $T/eval_${arm}.log 2>&1
    done
  done
  sleep 120
done
