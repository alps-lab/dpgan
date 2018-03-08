#!/usr/bin/env bash
set -x;
set -e;
for x in 1000 2000 3000 4000
    do
        python src/quality/lsun_bedroom_quality_fake.py ../dpgannew_outputs/19/lsun.bedroom.nodp.models.19/model-23693 --data-dir ~/Downloads/lsun/decompressed/bedroom_train_lmdb/ \
        -x $x --save-path lsun_bedroom_nodp_$x.npz --batch-size 100 --steps 2000;

        python src/quality/lsun_bedroom_quality_fake.py ../dpgannew_outputs/35/lsun.10cats.test.models/model-14500 --data-dir ~/Downloads/lsun/decompressed/bedroom_train_lmdb/ \
        -x $x --save-path lsun_bedroom_dp_$x.npz --batch-size 100 --steps 2000;
done;
