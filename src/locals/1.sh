#!/usr/bin/env bash
# identifier 1, run on uopod
python src/dp_mnist.py -a --data-dir ./data/mnist_data/ --save-dir ./outputs/mnist.new.e2.d1e-4.basic.c4.models \
--image-dir ./outputs/mnist.new.e2.d1e-4.basic.c4.images \
--epsilon 2.0 --delta 1e-4 --target-epsilons 2.0 \
--target-deltas 1e-4 --log-path ./logs/mnist.new.e2.d1e-4.basic.c4.log \
--batch-size 96 -g 4 --num-epoch 10 --num-critic 4 --save-every 200 --learning-rate 1e-3

python src/dp_lsun_bedroom.py -a --data-dir ./data/mnist_data/ --save-dir ./outputs/mnist.new.e2.d1e-4.basic.c4.models \
--image-dir ./outputs/mnist.new.e2.d1e-4.basic.c4.images \
--epsilon 2.0 --delta 1e-4 --target-epsilons 2.0 \
--target-deltas 1e-4 --log-path ./logs/mnist.new.e2.d1e-4.basic.c4.log \
--batch-size 96 -g 4 --num-epoch 10 --num-critic 4 --save-every 200