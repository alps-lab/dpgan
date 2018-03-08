#!/usr/bin/env bash
set -x;
set -e;
for x in 100 200 400 500 1000
    do
        python src/quality/celeba_quality_real.py -d ~/Dataset/CelebA/img_align_celeba_png/ --save-path \
            ./celeba_real_$x.npz \
            -x $x --batch-size 100 --steps 2000

done;

