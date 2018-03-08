#!/usr/bin/env bash
set -x;
set -e;

## no optimization at all
#python src/dp/dp_lsun_5cat.py -a /home/xinyang/Datasets/dpgan/lsun_5cats/private/ --save-dir \
#./outputs/optim.lsun.5cats.dp.1.models --epsilon 10.0 --delta 1e-4 --target-epsilons 10.0 \
#--target-deltas 1e-5 --moment 13 \
#--batch-size 64 -g 4 --num-epoch 3 --num-critic 4 --save-every 500 \
#--image-every 20 --gen-learning-rate 2e-4 --terminate \
#--adaptive-rate --clipper basic --sample-dir /home/xinyang/Datasets/dpgan/lsun_5cats/public/
#
## weight bias grouping
#python src/dp/dp_lsun_5cat.py -a /home/xinyang/Datasets/dpgan/lsun_5cats/private/ --save-dir \
#./outputs/optim.lsun.5cats.dp.2.models --epsilon 10.0 --delta 1e-4 --target-epsilons 10.0 \
#--target-deltas 1e-5 --moment 13 \
#--batch-size 64 -g 4 --num-epoch 3 --num-critic 4 --save-every 500 \
#--image-every 20 --gen-learning-rate 2e-4 --terminate \
#--adaptive-rate --clipper lsun --sample-dir /home/xinyang/Datasets/dpgan/lsun_5cats/public/
#
## automatic grouping



# estimation
python src/dp/dp_lsun_5cat.py -a /home/xinyang/Datasets/dpgan/lsun_5cats/private/ --save-dir \
./outputs/optim.lsun.5cats.dp.4.models --epsilon 10.0 --delta 1e-4 --target-epsilons 10.0 \
--target-deltas 1e-5 --moment 13 \
--batch-size 64 -g 4 --num-epoch 3 --num-critic 4 --save-every 500 \
--image-every 20 --gen-learning-rate 2e-4 --terminate \
--adaptive-rate --clipper lsun_est_simple \
--sample-dir /home/xinyang/Datasets/dpgan/lsun_5cats/public/ -e 2

# estimation + weight/bias grouping

# estimation + auto grouping
python src/dp/dp_lsun_5cat.py -a /home/xinyang/Datasets/dpgan/lsun_5cats/private/ --save-dir \
./outputs/optim.lsun.5cats.dp.6.models --epsilon 10.0 --delta 1e-4 --target-epsilons 10.0 \
--target-deltas 1e-5 --moment 13 \
--batch-size 64 -g 4 --num-epoch 3 --num-critic 4 --save-every 500 \
--image-every 20 --gen-learning-rate 2e-4 --terminate -e 2 \
--adaptive-rate --clipper lsun_ag_7 --sample-dir /home/xinyang/Datasets/dpgan/lsun_5cats/public/