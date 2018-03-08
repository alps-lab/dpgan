#!/usr/bin/env python
import argparse
import os
from itertools import chain

from sklearn.model_selection import ParameterGrid
import tensorflow as tf

from models.tasks.semi_mnist_attn import image_classifier_forward
from tasks.semi_classify_code_attn import run_task_eval
from utils.data_utils import MNISTLoader

NAME_STYLE = "start_%(gen_start)d_pub_%(public_num)d_final_%(gen_frac_final).1f_step_%(gen_frac_step)d"

GRID1 = ParameterGrid({
        "gen_start": [800, 1600],
        "gen_frac_init": [0.00],
        "gen_frac_final": [0.1, 0.2, 0.3],
        "gen_frac_step": [400, 800],
        "public_num": [100, 200, 300],
    }
)

GRID2 = ParameterGrid({
        "gen_start": [0],
        "gen_frac_init": [0.00],
        "gen_frac_final": [0.0],
        "gen_frac_step": [0],
        "public_num": [100, 200, 300, 500, 1000],
    }
)

GRID3 = ParameterGrid({
        "gen_start": [800, 1600],
        "gen_frac_init": [0.00],
        "gen_frac_final": [0.1, 0.2, 0.3],
        "gen_frac_step": [400, 800],
        "public_num": [500, 1000],
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

    for params in chain(GRID1, GRID2, GRID3):
        for key, value in params.items():
            setattr(config, key, value)
        name = NAME_STYLE % params
        print("config: %r" % config)
        print("resetting environment...")
        tf.reset_default_graph()

        eval_data_loader = MNISTLoader(config.data_dir, include_test=True, include_train=False)

        mean_accuracy = run_task_eval(config,
                          eval_data_loader,
                          image_classifier_forward,
                          model_dir=os.path.join(model_dir, name + "_models"))
        if fobj is not None:
            fobj.write("%s: %.4f\n" % (name, mean_accuracy))
        else:
            print("%s: %.4f\n" % (name, mean_accuracy))

    if fobj is not None:
        fobj.close()


