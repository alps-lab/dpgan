#!/usr/bin/env bash
set -x;
set -e;

# no optimization at all
python src/dp/dp_mnist.py -a --epsilon 4.0 --delta 1e-4 --target-epsilon 4.0 --target-delta 1e-5 \
--data-dir ./data/mnist_data --sample-ratio 0.02 -g 4 --clipper basic \
--save-dir ./outputs/optim.mnist.dp.1.models --batch-size 64 --save-every 500 \
--adaptive-rate --num-critic 4 --exclude-test --terminate;

# weight bias grouping
python src/dp/dp_mnist.py -a --epsilon 4.0 --delta 1e-4 --target-epsilon 4.0 --target-delta 1e-5 \
--data-dir ./data/mnist_data --sample-ratio 0.02 -g 4 --clipper mnist \
--save-dir ./outputs/optim.mnist.dp.2.models --batch-size 64 --save-every 500 \
--adaptive-rate --num-critic 4 --exclude-test --terminate;

## automatic grouping
#python src/dp/dp_mnist.py -a --epsilon 4.0 --delta 1e-4 --target-epsilon 4.0 --target-delta 1e-5 \
#--data-dir ./data/mnist_data --sample-ratio 0.02 -g 4 --clipper mnist_est \
#--save-dir ./outputs/optim.mnist.dp.3.models --batch-size 64 --save-every 500 \
#--adaptive-rate --num-critic 4 --exclude-test

# estimation
python src/dp/dp_mnist.py -a --epsilon 4.0 --delta 1e-4 --target-epsilon 4.0 --target-delta 1e-5 \
--data-dir ./data/mnist_data --sample-ratio 0.02 -g 4 --clipper mnist_est_simple \
--save-dir ./outputs/optim.mnist.dp.4.models --batch-size 64 --save-every 500 \
--adaptive-rate --num-critic 4 --exclude-test --terminate;

## estimation + weight/bias grouping
#python src/dp/dp_mnist.py -a --epsilon 4.0 --delta 1e-4 --target-epsilon 4.0 --target-delta 1e-5 \
#--data-dir ./data/mnist_data --sample-ratio 0.02 -g 4 --clipper mnist_est \
#--save-dir ./outputs/optim.mnist.dp.5.models --batch-size 64 --save-every 500 \
#--adaptive-rate --num-critic 4 --exclude-test

# estimation + auto grouping
python src/dp/dp_mnist.py -a --epsilon 4.0 --delta 1e-4 --target-epsilon 4.0 --target-delta 1e-5 \
--data-dir ./data/mnist_data --sample-ratio 0.02 -g 4 --clipper mnist_ag_5 \
--save-dir ./outputs/optim.mnist.dp.6.models --batch-size 64 --save-every 500 \
--adaptive-rate --num-critic 4 --exclude-test --terminate;