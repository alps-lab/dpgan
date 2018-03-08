#!/usr/bin/env python
import argparse
import os
from itertools import chain

from sklearn.model_selection import ParameterGrid
import tensorflow as tf

from models.tasks.semi_lsun_5cat_attn import image_classifier_forward
from tasks.semi_classify_code_attn import run_task_eval
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
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", metavar="MODEL_DIR")
    parser.add_argument("eval_data_dir", metavar="EVAL_DATA_DIR")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=100)
    parser.add_argument("--save-path", dest="save_path")
    parser.add_argument("--dim", dest="gen_dim", default=64, type=int)

    config = parser.parse_args()
    model_dir = config.model_dir

    if config.save_path is not None:
        fobj = open(config.save_path, "w")
    else:
        fobj = None

    for params in chain(GRID1, GRID2):
        for key, value in params.items():
            setattr(config, key, value)
        name = NAME_STYLE % params
        print("config: %r" % config)
        print("resetting environment...")
        tf.reset_default_graph()

        eval_data_loader = LSUNCatLoader(get_lsun_patterns(config.eval_data_dir),
                                         num_workers=5, block_size=20,
                                         actions=lsun_process_actions())
        try:
            eval_data_loader.start_fetch()
            mean_accuracy = run_task_eval(config,
                              eval_data_loader,
                              image_classifier_forward,
                              model_dir=os.path.join(model_dir, name + "_models"))
            if fobj is not None:
                fobj.write("%s: %.4f\n" % (name, mean_accuracy))
                print("fuck")

        finally:
            eval_data_loader.stop_fetch()

    if fobj is not None:
        fobj.close()


