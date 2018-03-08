#!/usr/bin/env python
import argparse
import os
from itertools import chain

from sklearn.model_selection import ParameterGrid
import tensorflow as tf

from models.gans.d64_resnet_dcgan import generator_forward
from models.tasks.semi_lsun_5cat_attn import image_classifier_forward, code_classifier_forward
from tasks.semi_classify_code_attn import run_task
from utils.data_mp import LSUNCatLoader, lsun_process_actions, get_lsun_patterns

NAME_STYLE = "start_%(gen_start)d_pub_%(public_num)d_final_%(gen_frac_final).1f_step_%(gen_frac_step)d"


GRID1 = ParameterGrid({
        "gen_start": [4000],
        "gen_frac_init": [0.00],
        "gen_frac_final": [0.2, 0.3, 0.4],
        "gen_frac_step": [2000],
        "public_num": [5000, 15000, 25000, 50000],
    }
)

GRID2 = ParameterGrid({
        "gen_start": [0],
        "gen_frac_init": [0.00],
        "gen_frac_final": [0.0],
        "gen_frac_step": [0],
        "public_num": [5000, 15000, 25000, 50000],
    }
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", metavar="TRAIN_DATA_DIR")
    parser.add_argument("eval_data_dir", metavar="EVAL_DATA_DIR")
    parser.add_argument("model_path", metavar="MODELPATH")

    parser.add_argument("--log-dir", dest="log_dir")
    parser.add_argument("--save-dir", dest="save_dir")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=100)
    parser.add_argument("-e", "--num-epoch", dest="num_epoch", type=int, default=1)
    parser.add_argument("--dim", dest="gen_dim", default=64, type=int)
    parser.add_argument("--public-num", dest="public_num", type=int, default=50000)
    parser.add_argument("--public-seed", dest="public_seed", type=int, default=1024)
    parser.add_argument("-n", "--num-example", dest="num_example", type=int, default=1000000)

    config = parser.parse_args()
    save_dir = config.save_dir
    log_dir = config.log_dir
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    for params in chain(GRID1, GRID2):
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

        train_data_loader = LSUNCatLoader(get_lsun_patterns(config.train_data_dir), num_workers=10,
                                          block_size=20,
                                          max_blocks=500,
                                          max_numbers=None,
                                          actions=lsun_process_actions(),
                                          public_num=config.public_num,
                                          public_seed=1024)
        eval_data_loader = LSUNCatLoader(
            get_lsun_patterns(config.eval_data_dir), block_size=20, max_numbers=None,
            actions=lsun_process_actions(), num_workers=4)

        try:
            train_data_loader.start_fetch()
            eval_data_loader.start_fetch()

            run_task(config, train_data_loader, eval_data_loader,
                     generator_forward, code_classifier_forward,
                     image_classifier_forward,
                     image_classifier_optimizer=tf.train.AdamOptimizer(),
                     code_classifier_optimizer=tf.train.AdamOptimizer(),
                     model_path=config.model_path)
            print("done")
        finally:
            train_data_loader.stop_fetch()
            eval_data_loader.stop_fetch()
