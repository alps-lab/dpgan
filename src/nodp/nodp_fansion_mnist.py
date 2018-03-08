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

    config = parser.parse_args()

    np.random.seed()

    data_loader = MNISTLoader(config.data_dir)

    gen_optimizer = tf.train.AdamOptimizer(config.gen_learning_rate, beta1=0.5, beta2=0.9)
    disc_optimizer = tf.train.AdamOptimizer(config.learning_rate, beta1=0.5, beta2=0.9)

    train(config, data_loader, mnist.generator_forward, mnist.discriminator_forward,
          gen_optimizer=gen_optimizer,
          disc_optimizer=disc_optimizer)