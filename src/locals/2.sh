#!/usr/bin/env bash
# 4 - LSUN-bedroom - Gaussian moment
python src/dp_lsun_bedroom.py -a /home/xinyang/Downloads/lsun_decompressed/bedroom_train_lmdb \
--save-dir ./outputs/lsun.new.e10.d1e-4.basic.c4.models \
--image-dir ./outputs/lsun.new.e10.d1e-4.basic.c4.images \
--epsilon 10.0 --delta 1e-4 --target-epsilons 10.0 \
--target-deltas 1e-4 --log-path ./logs/lsun.new.e10.d1e-4.basic.c4.log --moment 15 \
--batch-size 48 -g 4 --num-epoch 3 --num-critic 4 --save-every 400 --image-every 30