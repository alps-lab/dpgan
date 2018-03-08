#!/usr/bin/env bash
set -x;
set -e;

# CelebA (48 * 48)
## no optimization at all

#python src/dp/dp_celeba_48.py -a -b 48 -e 10 \
#-c 4 --save-dir ./outputs/optim.celeba48.dp.1.models \
#--save-every 500 -g 4 --epsilon 8.0 --delta 1e-4 \
#--target-epsilons 10.0 --target-deltas 1e-5 \
#--clipper basic --moment 13 --terminate --adaptive-rate \
#--sample-dir /home/xinyang/Datasets/dpgan/CelebA/public \
#/home/xinyang/Datasets/dpgan/CelebA/private

# weight bias grouping
#python src/dp/dp_celeba_48.py -a -b 48 -e 10 \
#-c 4 --save-dir ./outputs/optim.celeba48.dp.2.models \
#--save-every 500 -g 4 --epsilon 8.0 --delta 1e-4 \
#--target-epsilons 10.0 --target-deltas 1e-5 \
#--clipper celeba_48 --moment 13 --terminate --adaptive-rate \
#--sample-dir /home/xinyang/Datasets/dpgan/CelebA/public \
#/home/xinyang/Datasets/dpgan/CelebA/private

## estimation
#python src/dp/dp_celeba_48.py -a -b 48 -e 10 \
#-c 4 --save-dir ./outputs/optim.celeba48.dp.4.models \
#--save-every 200 -g 4 --epsilon 8.0 --delta 1e-4 \
#--target-epsilons 10.0 --target-deltas 1e-5 \
#--clipper celeba_48_est_simple --moment 13 --terminate --adaptive-rate \
#--sample-dir /home/xinyang/Datasets/dpgan/CelebA/public \
#/home/xinyang/Datasets/dpgan/CelebA/private;

# estimation + auto grouping
python src/dp/dp_celeba_48.py -a -b 48 -e 10 \
-c 4 --save-dir ./outputs/optim.celeba48.dp.7.models \
--save-every 200 -g 4 --epsilon 8.0 --delta 1e-4 \
--target-epsilons 10.0 --target-deltas 1e-5 \
--clipper celeba_48_ag_6 --moment 13 --terminate --adaptive-rate \
--sample-dir /home/xinyang/Datasets/dpgan/CelebA/public \
/home/xinyang/Datasets/dpgan/CelebA/private;