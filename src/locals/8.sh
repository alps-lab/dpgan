#!/usr/bin/env bash
set -x;
set -e;
for x in 100 200 400 500 1000
    do
        python src/quality/celeba_quality_fake.py -d ~/Dataset/CelebA/img_align_celeba_png/ --save-path \
            ./celeba_nodp_$x.npz \
            ../dpgannew_outputs/70/celeba.48.nodp.models/model-15819 -x $x --batch-size 100 --steps 2000

        python src/quality/celeba_quality_fake.py -d ~/Dataset/CelebA/img_align_celeba_png/ --save-path \
            ./celeba_dp_$x.npz \
            ../dpgannew_outputs/69/celeba_48_test_models/model-3400 -x $x --batch-size 100 --steps 2000
done;

