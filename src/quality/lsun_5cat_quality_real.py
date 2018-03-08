#!/usr/bin/env python
from six.moves import xrange, cPickle

import tempfile
import os
import argparse

import numpy as np

from utils.data_mp import LSUNCatLoader, get_lsun_patterns, lsun_process_actions
from tasks.lsun_5cats_resnet18_score import get_resnet18_score


def images_iter(names):
    for i, name in enumerate(names):
        images = np.load(name)
        images = list(images)

        for img in images:
            yield img


def main(config):
    num_step = config.num_step
    data_loader = LSUNCatLoader(get_lsun_patterns(config.data_dir),
                                num_workers=4,
                                actions=lsun_process_actions())

    names = []
    fobjs = []
    try:
        data_loader.start_fetch()
        print("generating images...")
        for _ in xrange(num_step):
            fd, name = tempfile.mkstemp(suffix=".npy")
            fobj = os.fdopen(fd, "wb+")
            names.append(name)
            fobjs.append(fobj)
            image_arr = data_loader.next_batch(config.batch_size)[0]
            np.save(fobj, image_arr, allow_pickle=False)
            fobj.close()

        mean_score, std_score = get_resnet18_score(images_iter(names),
                                                config.model_path,
                                                batch_size=100,
                                                split=10)

        print("mean = %.4f, std = %.4f." % (mean_score, std_score))

        if config.save_path is not None:
            with open(config.save_path, "wb") as f:
                cPickle.dump(dict(batch_size=config.batch_size,
                                  scores=dict(mean=mean_score, std=std_score)), f)
    finally:
        data_loader.stop_fetch()
        for name in names:
            os.unlink(name)
        for fobj in fobjs:
            fobj.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", metavar="DATADIR")
    parser.add_argument("model_path", metavar="MODELPATH")
    parser.add_argument("--steps", dest="num_step",type=int, default=2000)
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=100)
    parser.add_argument("--save-path", dest="save_path")
    main(parser.parse_args())
