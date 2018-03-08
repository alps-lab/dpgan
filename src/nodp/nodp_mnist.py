#!/usr/bin/env python
import numpy as np
import tensorflow as tf

from nodp.train import train
from models.gans import mnist
from utils.data_utils import MNISTLoader
from utils.parsers import create_nodp_parser

if __name__ == "__main__":
    parser = create_nodp_parser()
    parser.add_argument("--dim", default=64, type=int, dest="dim")
    parser.add_argument("--data-dir", default="./data/mnist_data", dest="data_dir")
    parser.add_argument("--learning-rate", default=4e-4, type=float, dest="learning_rate")
    parser.add_argument("--gen-learning-rate", default=4e-4, type=float, dest="gen_learning_rate")
    parser.add_argument("--sample-seed", dest="sample_seed", type=int, default=1024)
    parser.add_argument("--sample-ratio", dest="sample_ratio", type=float)
    parser.add_argument("--exclude-train", dest="exclude_train", action="store_true")
    parser.add_argument("--exclude-test", dest="exclude_test", action="store_true")

    config = parser.parse_args()

    np.random.seed()

    if config.sample_ratio is None:
        data_loader = MNISTLoader(config.data_dir)
    else:
        data_loader = MNISTLoader(config.data_dir, include_train=not config.exclude_train,
                                  include_test=not config.exclude_test,
                                  last=int(50000 * config.sample_ratio),
                                  seed=config.sample_seed
                                )


    gen_optimizer = tf.train.AdamOptimizer(config.gen_learning_rate, beta1=0.5, beta2=0.9)
    disc_optimizer = tf.train.AdamOptimizer(config.learning_rate, beta1=0.5, beta2=0.9)

    train(config, data_loader, mnist.generator_forward, mnist.discriminator_forward,
          gen_optimizer=gen_optimizer,
          disc_optimizer=disc_optimizer)