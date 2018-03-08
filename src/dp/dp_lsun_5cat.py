#!/usr/bin/env python
import numpy as np
import tensorflow as tf

from dp.train import train
from models.gans import d64_resnet_dcgan
from utils.accounting import GaussianMomentsAccountant
from utils.data_mp import LSUNCatLoader, lsun_process_actions, get_lsun_patterns
from utils.parsers import create_dp_parser
from utils.clippers import get_clipper
from utils.schedulers import get_scheduler
from dp.supervisors.basic_lsun_dummy import BasicSupervisorLSUN

if __name__ == "__main__":
    parser = create_dp_parser()
    parser.add_argument("--gen-dim", default=64, type=int, dest="gen_dim")
    parser.add_argument("--disc-dim", default=64, type=int, dest="disc_dim")
    parser.add_argument("--learning-rate", default=2e-4, type=float, dest="learning_rate")
    parser.add_argument("--gen-learning-rate", default=2e-4, type=float, dest="gen_learning_rate")
    parser.add_argument("--beta1", default=0., type=float, dest="beta1")
    parser.add_argument("--beta2", default=0.9, type=float, dest="beta2")
    parser.add_argument("data_dir", metavar="DATADIR")
    parser.add_argument("--image-size", default=64, type=int, dest="image_size")
    parser.add_argument("--adaptive-rate", dest="adaptive_rate", action="store_true")
    parser.add_argument("--no-noise", dest="no_noise", action="store_true")
    parser.add_argument("--sample-dir", dest="sample_dir")

    config = parser.parse_args()

    np.random.seed()
    if config.enable_accounting:
        config.sigma = np.sqrt(2.0 * np.log(1.25 / config.delta)) / config.epsilon
        print("Now with new sigma: %.4f" % config.sigma)

    if config.image_size == 64:
        patterns = get_lsun_patterns(config.data_dir)
        print(patterns)
        data_loader = LSUNCatLoader(patterns, num_workers=4, actions=lsun_process_actions(),
                                    block_size=16, max_blocks=256)
        data_loader.start_fetch()
        generator_forward = d64_resnet_dcgan.generator_forward
        discriminator_forward = d64_resnet_dcgan.discriminator_forward
    else:
        raise NotImplementedError("Unsupported image size %d." % config.image_size)

    if config.enable_accounting:
        accountant = GaussianMomentsAccountant(data_loader.num_steps(1), config.moment)
        if config.log_path:
            open(config.log_path, "w").close()
    else:
        accountant = None

    if config.adaptive_rate:
        lr = tf.placeholder(tf.float32, shape=())
    else:
        lr = config.learning_rate

    gen_optimizer = tf.train.AdamOptimizer(config.gen_learning_rate, beta1=config.beta1,
                                       beta2=config.beta2)
    disc_optimizer = tf.train.AdamOptimizer(lr, beta1=config.beta1,
                                       beta2=config.beta2)

    clipper_ret = get_clipper(config.clipper, config)
    if isinstance(clipper_ret, tuple):
        clipper, sampler = clipper_ret
        patterns = get_lsun_patterns(config.sample_dir)
        print(patterns)
        sampler.set_data_loader(LSUNCatLoader(patterns, block_size=16, max_blocks=64,
                                              num_workers=2, actions=lsun_process_actions()))
        sampler.data_loader.start_fetch()
        sampler.keep_memory = False
    else:
        clipper = clipper_ret
        sampler = None
    scheduler = get_scheduler(config.scheduler, config)

    def callback_before_train(_0, _1, _2):
        print(clipper.info())

    supervisor = BasicSupervisorLSUN(config, clipper, scheduler,
                                     sampler=sampler, callback_before_train=callback_before_train)
    if config.adaptive_rate:
        supervisor.put_key("lr", lr)

    try:
        train(config, data_loader, generator_forward, discriminator_forward,
              disc_optimizer=disc_optimizer,
              gen_optimizer=gen_optimizer, accountant=accountant, supervisor=supervisor)
    finally:
        data_loader.stop_fetch()
        if sampler is not None:
            sampler.data_loader.stop_fetch()
