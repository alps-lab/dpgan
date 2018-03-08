#!/usr/bin/env bash
set -x;
set -e;
for x in 1000 2000 3000 4000
    do
        python src/quality/lsun_bedroom_quality_real.py --data-dir ~/Downloads/lsun/decompressed/bedroom_train_lmdb/ \
        -x $x --save-path lsun_bedroom_real_$x.npz --batch-size 100 --steps 2000;

done;
