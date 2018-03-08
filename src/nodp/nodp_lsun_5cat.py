#!/usr/bin/env python
import sys

import numpy as np
import tensorflow as tf

from nodp.train import train
from models.gans import d64_resnet_dcgan
from utils.data_mp import LSUNCatLoader, lsun_process_actions, get_lsun_patterns
from utils.parsers import create_nodp_parser


if __name__ == "__main__":
    parser = create_nodp_parser()
    parser.add_argument("--gen-dim", default=64, type=int, dest="gen_dim")
    parser.add_argument("--disc-dim", default=64, type=int, dest="disc_dim")
    parser.add_argument("--learning-rate", default=2e-4, type=float, dest="learning_rate")
    parser.add_argument("--gen-learning-rate", default=2e-4, type=float, dest="gen_learning_rate")
    parser.add_argument("--beta1", default=0., type=float, dest="beta1")
    parser.add_argument("--beta2", default=0.9, type=float, dest="beta2")
    parser.add_argument("data_dir", metavar="DATADIR")

    config = parser.parse_args()
    np.random.seed()

    patterns = get_lsun_patterns(config.data_dir)
    # print(patterns)
    data_loader = LSUNCatLoader(patterns, num_workers=4, actions=lsun_process_actions(),
                                block_size=16, max_blocks=256)
    generator_forward = d64_resnet_dcgan.generator_forward
    discriminator_forward = d64_resnet_dcgan.discriminator_forward

    gen_optimizer = tf.train.AdamOptimizer(config.gen_learning_rate, beta1=config.beta1,
                                       beta2=config.beta2)
    disc_optimizer = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1,
                                       beta2=config.beta2)

    try:
        data_loader.start_fetch()
        train(config, data_loader, generator_forward, discriminator_forward,
              disc_optimizer=disc_optimizer,
              gen_optimizer=gen_optimizer)
    except KeyboardInterrupt:
        print("Interrupted...", file=sys.stderr)
        raise
    finally:
        data_loader.stop_fetch()
