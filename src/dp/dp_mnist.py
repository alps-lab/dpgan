#!/usr/bin/env python
import numpy as np
import tensorflow as tf

from dp.train import train
from models.gans import mnist
from utils.accounting import GaussianMomentsAccountant
from utils.data_utils import MNISTLoader
from utils.parsers import create_dp_parser
from utils.clippers import get_clipper
from utils.schedulers import get_scheduler
from dp.supervisors.basic_mnist import BasicSupervisorMNIST


if __name__ == "__main__":
    parser = create_dp_parser()
    parser.add_argument("--dim", default=64, type=int, dest="dim")
    parser.add_argument("--data-dir", default="./data/mnist_data", dest="data_dir")
    parser.add_argument("--learning-rate", default=2e-4, type=float, dest="learning_rate")
    parser.add_argument("--gen-learning-rate", default=2e-4, type=float, dest="gen_learning_rate")
    parser.add_argument("--adaptive-rate", dest="adaptive_rate", action="store_true")
    parser.add_argument("--sample-seed", dest="sample_seed", type=int, default=1024)
    parser.add_argument("--sample-ratio", dest="sample_ratio", type=float)
    parser.add_argument("--exclude-train", dest="exclude_train", action="store_true")
    parser.add_argument("--exclude-test", dest="exclude_test", action="store_true")

    config = parser.parse_args()
    config.dataset = "mnist"

    np.random.seed()
    if config.enable_accounting:
        config.sigma = np.sqrt(2.0 * np.log(1.25 / config.delta)) / config.epsilon
        print("Now with new sigma: %.4f" % config.sigma)

    if config.sample_ratio is not None:
        kwargs = {}
        gan_data_loader = MNISTLoader(config.data_dir, include_train=not config.exclude_train,
                                  include_test=not config.exclude_test,
                                  first=int(50000 * (1 - config.sample_ratio)),
                                  seed=config.sample_seed
                                )
        sample_data_loader = MNISTLoader(config.data_dir, include_train=not config.exclude_train,
                                  include_test=not config.exclude_test,
                                  last=int(50000 * config.sample_ratio),
                                  seed=config.sample_seed
                                )
    else:
        gan_data_loader = MNISTLoader(config.data_dir, include_train=not config.exclude_train,
                                  include_test=not config.exclude_test)

    if config.enable_accounting:
        accountant = GaussianMomentsAccountant(gan_data_loader.n, config.moment)
        if config.log_path:
            open(config.log_path, "w").close()
    else:
        accountant = None

    if config.adaptive_rate:
        lr = tf.placeholder(tf.float32, shape=())
    else:
        lr = config.learning_rate

    gen_optimizer = tf.train.AdamOptimizer(config.gen_learning_rate, beta1=0.5, beta2=0.9)
    disc_optimizer = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)

    clipper_ret = get_clipper(config.clipper, config)
    if isinstance(clipper_ret, tuple):
        clipper, sampler = clipper_ret
        sampler.set_data_loader(sample_data_loader)
        sampler.keep_memory = False
    else:
        clipper = clipper_ret
        sampler = None

    scheduler = get_scheduler(config.scheduler, config)
    def callback_before_train(_0, _1, _2):
        print(clipper.info())
    supervisor = BasicSupervisorMNIST(config, clipper, scheduler, sampler=sampler,
                                      callback_before_train=callback_before_train)
    if config.adaptive_rate:
        supervisor.put_key("lr", lr)

    train(config, gan_data_loader, mnist.generator_forward, mnist.discriminator_forward,
          gen_optimizer=gen_optimizer,
          disc_optimizer=disc_optimizer, accountant=accountant,
          supervisor=supervisor)