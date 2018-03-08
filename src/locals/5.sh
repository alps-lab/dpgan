#!/usr/bin/env bash
python src/dp_mnist_v2.py -a --data-dir ./data/mnist_data/ --save-dir ./outputs/mnist.new.e2.d1e-4.basic.c4.cl1.models \
--image-dir ./outputs/mnist.new.e2.d1e-4.basic.c4.cl1.images \
--epsilon 2.0 --delta 1e-4 --target-epsilons 2.0 \
--target-deltas 1e-4 --log-path ./logs/mnist.new.e2.d1e-4.basic.c4.cl1.log \
--batch-size 96 -g 4 --num-epoch 6 --num-critic 4 --save-every 200 \
--learning-rate 1e-3 --gen-learning-rate 5e-4 --clipper c1


python src/dp_mnist_v2.py -a --data-dir ./data/mnist_data/ --save-dir ./outputs/mnist.new.e2.d1e-4.basic.c4.s1.models \
--image-dir ./outputs/mnist.new.e2.d1e-4.basic.c4.s1.images \
--epsilon 2.0 --delta 1e-4 --target-epsilons 2.0 \
--target-deltas 1e-4 --log-path ./logs/mnist.new.e2.d1e-4.basic.c4.s1.log \
--batch-size 96 -g 4 --num-epoch 6 --num-critic 4 --save-every 200 \
--learning-rate 1e-3 --gen-learning-rate 5e-4 --scheduler 1

