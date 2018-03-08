#!/usr/bin/env python
import argparse
import random
import os
import glob


CATEGORIES_TO_SAMPLE = {
    "bedroom_train_lmdb": 500000,
    "kitchen_train_lmdb": 500000,
    "bridge_train_lmdb": 500000,
    "living_room_train_lmdb": 500000,
    "tower_train_lmdb": 500000,
}


def sample(config):
    random_obj = random.Random(config.seed)
    for subdir, n in sorted(CATEGORIES_TO_SAMPLE.items(), key=lambda x: x[0]):
        paths = glob.glob(os.path.join(config.source_dir, subdir, "*.jpg"))
        random_obj.shuffle(paths)
        paths = paths[:n]
        os.makedirs(os.path.join(config.target_dir, subdir))

        for path in paths:
            basename = os.path.basename(path)
            os.link(path, os.path.join(config.target_dir, subdir, basename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", metavar="SOURCEDIR")
    parser.add_argument("target_dir", metavar="TARGETDIR")
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=1024)

    sample(parser.parse_args())
