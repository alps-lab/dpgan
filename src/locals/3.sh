#!/usr/bin/env bash
# (17, ipod)
python src/nodp_mnist.py -b 72 -e 4 -c 5 --image-dir ./outputs/mnist.nodp.images --save-dir ./outputs/mnist.nodp.models -g 2 --image-every 100 --data-dir ./data/mnist_data --save-every 400

# (18)
python src/nodp_mnist.py -b 72 -e 4 -c 5 --image-dir ./outputs/mnist.nodp.images --save-dir ./outputs/mnist.nodp.models -g 2 --image-every 100 --data-dir ./data/mnist_data --save-every 400 --learning-rate 1e-4 --gen-learning-rate 1e-4

# (19)
python src/nodp_lsun_bedroom.py -b 72 -e 1 -c 4 --image-dir ./outputs/lsun.bedroom.nodp.images.19 --save-dir \
./outputs/lsun.bedroom.nodp.models.19 --learning-rate 2e-4 --gen-learning-rate 2e-4 -g 2 --save-every 1000 --image-every 20 \
/home/xinyang/Downloads/lsun_decompressed/bedroom_train_lmdb

# (20)
python src/nodp_lsun_bedroom.py -b 64 -e 1 -c 5 --image-dir ./outputs/lsun.bedroom.nodp.images.20 --save-dir \
./outputs/lsun.bedroom.nodp.models.20 --learning-rate 1e-4 --gen-learning-rate 1e-4 -g 4 --save-every 1000 --image-every 100 \
/home/xinyang/Downloads/lsun_decompressed/bedroom_train_lmdb
