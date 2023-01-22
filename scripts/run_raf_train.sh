#!/bin/sh

OUT_FILE="../assets/raf_trained_weights.dat"

rm -f $OUT_FILE
raf train $(ls ../assets/train_data/*.fa -A1) 2>&1 \
  | grep -E "^Epoch" | tail -1 | sed -E -n "s/^.*w = \[(.*)\].*$/\1/p" > $OUT_FILE
