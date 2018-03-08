#!/usr/bin/env bash
python src/nodp/nodp_mnist.py -b 64 -e 1000 -c 5 \
--save-dir ./outputs/mnist.nodp.77.models -g 2 \
--data-dir ./data/mnist_data \
--save-every 50 --learning-rate 2e-4 \
--image-every 20 --image-dir ./outputs/mnist.nodp.77.images \
--gen-learning-rate 2e-4 --sample-ratio 0.02 \
--exclude-test --total-step 600

python src/nodp/nodp_celeba_48.py -b 64 -e 1000 -c 4 \
--image-dir ./outputs/celeba.nodp.78.images \
--save-dir ./outputs/celeba.nodp.78.models -g 2 \
--image-every 20 --save-every 100 \
~/Dataset/CelebA/public --total-step 1500


python src/nodp/nodp_lsun_5cat.py -b 64 -e 1000 -c 4 \
--image-dir ./outputs/lsun.5cats.nodp.79.images \
--save-dir ./outputs/lsun.5cats.nodp.79.models \
--save-every 400 --image-every 25 \
/home/xinyang/Datasets/dpgan/lsun_5cats/public/ \
--total-step 4000 -g 2

python src/nodp/nodp_lsun_bedroom.py -b 64 -e 1000 -c 4 \
--image-dir ./outputs/lsun.bedroom.nodp.images.80 \
--save-dir ./outputs/lsun.bedroom.nodp.models.80 --learning-rate 2e-4 \
--gen-learning-rate 2e-4 -g 2 --save-every 400 --image-every 25 \
/home/xinyang/Dataset/dpgan/lsun_bedroom/public/ \
--total-step 4000