#!/usr/bin/env python
import argparse
import os

from sklearn.model_selection import ParameterGrid
import tensorflow as tf

from models.tasks.mnist
from tasks.semi_classify import run_task_eval
from utils.data_utils import MNISTLoader

NAME_STYLE = "pub_%(public_num)d_final_%(gen_frac_final).1f_step_%(gen_frac_step)d"

GRID = ParameterGrid({
        "gen_frac_init": [0.00],
        "gen_frac_final": [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0],
        "gen_frac_step": [0, 100, 200, 400, 500, 600, 800],
        "public_num": [100, 250, 500, 750, 1000],
    }
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", metavar="MODELDIR")
    parser.add_argument("--data-dir", dest="data_dir", default="/tmp/mnist")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=100)
    parser.add_argument("--save-path", dest="save_path")

    config = parser.parse_args()
    model_dir = config.model_dir

    if config.save_path is not None:
        fobj = open(config.save_path, "w")
    else:
        fobj = None

    for params in GRID:
        for key, value in params.items():
            setattr(config, key, value)
        name = NAME_STYLE % params
        print("config: %r" % config)
        print("resetting environment...")
        tf.reset_default_graph()

        eval_data_loader = MNISTLoader(config.data_dir, include_test=True, include_train=False)

        mean_accuracy = run_task_eval(config,
                          eval_data_loader,
                          classifier_forward,
                          model_dir=os.path.join(model_dir, name + "_models"))
        if fobj is not None:
            fobj.write("%s: %.4f\n" % (name, mean_accuracy))
            print("fuck")

    if fobj is not None:
        fobj.close()


