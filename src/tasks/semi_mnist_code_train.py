#!/usr/bin/env python
import argparse
import os

from sklearn.model_selection import ParameterGrid
import tensorflow as tf

from models.gans.mnist import generator_forward
from models.tasks.semi_mnist import image_classifier_forward, code_classifier_forward
from tasks.semi_classify_code import run_task
from utils.data_utils import MNISTLoader

NAME_STYLE = "start_%(gen_start)d_pub_%(public_num)d_final_%(gen_frac_final).1f_step_%(gen_frac_step)d"

num_images = 50000

GRID = ParameterGrid({
        "gen_start": [1600, 2400],
        "gen_frac_init": [0.00],
        "gen_frac_final": [0.1],
        "gen_frac_step": [400, 800],
        "public_num": [100, 250],
    }
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", metavar="MODELPATH")
    parser.add_argument("--log-dir", dest="log_dir")
    parser.add_argument("--save-dir", dest="save_dir")
    parser.add_argument("--data-dir", dest="data_dir", default="/tmp/mnist")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("-e", "--num-epoch", dest="num_epoch", type=int, default=9)
    parser.add_argument("--dim", dest="dim", default=64, type=int)
    parser.add_argument("--public-num", dest="public_num", type=int, default=50000)
    parser.add_argument("--public-seed", dest="public_seed", type=int, default=1024)
    parser.add_argument("-n", "--num-example", dest="num_example", type=int, default=50000)

    config = parser.parse_args()
    save_dir = config.save_dir
    log_dir = config.log_dir
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    for params in GRID:
        for key, value in params.items():
            setattr(config, key, value)
        name = NAME_STYLE % params
        if save_dir is not None:
            config.save_dir = os.path.join(save_dir, name + "_models")
            os.makedirs(config.save_dir, exist_ok=True)
        if log_dir is not None:
            config.log_path = os.path.join(log_dir, name + ".log")

        print("config: %r" % config)
        print("resetting environment...")
        tf.reset_default_graph()

        train_data_loader = MNISTLoader(config.data_dir, include_test=False,
                                        first=config.public_num,
                                        seed=config.public_seed)
        eval_data_loader = MNISTLoader(config.data_dir, include_test=True, include_train=False)

        run_task(config, train_data_loader, eval_data_loader,
                 generator_forward, code_classifier_forward,
                 image_classifier_forward,
                 image_classifier_optimizer=tf.train.AdamOptimizer(),
                 code_classifier_optimizer=tf.train.AdamOptimizer(),
                 model_path=config.model_path)

