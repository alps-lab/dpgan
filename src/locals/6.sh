#!/usr/bin/env bash
python src/dp/dp_mnist.py -a --epsilon 4.0 --delta 1e-4 --target-epsilon 4.0 --target-delta 1e-4 \
--data-dir ./data/mnist_data --sample-ratio 0.02 -g 4 --clipper mnist_est \
--log-path ./logs/mnist.test.log --image-dir ./outputs/mnist.test.images \
--save-dir ./outputs/mnist.test.models --batch-size 64 --save-every 500 \
--image-every 20 --adaptive-rate --num-critic 4