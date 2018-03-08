#!/usr/bin/env python
import numpy as np
import tensorflow as tf

from nodp.train import train
from models.gans import d64_resnet_dcgan
from utils.data_mp import LSUNLoader, lsun_process_actions
from utils.parsers import create_nodp_parser

if __name__ == "__main__":
    parser = create_nodp_parser()
    parser.add_argument("--gen-dim", default=64, type=int, dest="gen_dim")
    parser.add_argument("--disc-dim", default=64, type=int, dest="disc_dim")
    parser.add_argument("--learning-rate", default=2e-4, type=float, dest="learning_rate")
    parser.add_argument("--gen-learning-rate", default=2e-4, type=float, dest="gen_learning_rate")
    parser.add_argument("--beta1", default=0., type=float, dest="beta1")
    parser.add_argument("--beta2", default=0.9, type=float, dest="beta2")
    parser.add_argument("--image-size", default=64, type=int, dest="image_size", choices=[64, 32])
    parser.add_argument("data_dir", metavar="DATADIR")

    config = parser.parse_args()

    np.random.seed()
    if config.image_size == 64:
        data_loader = LSUNLoader(config.data_dir, actions=lsun_process_actions())
        generator_forward = d64_resnet_dcgan.generator_forward
        discriminator_forward = d64_resnet_dcgan.discriminator_forward
    else:
        raise NotImplementedError()
    data_loader.start_fetch()

    gen_optimizer = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1,
                                       beta2=config.beta2)
    disc_optimizer = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1,
                                       beta2=config.beta2)
    try:
        train(config, data_loader, generator_forward, discriminator_forward,
              disc_optimizer=disc_optimizer,
              gen_optimizer=gen_optimizer)
    except KeyboardInterrupt:
        print("Interrupted...")
        raise
    finally:
        data_loader.stop_fetch()
