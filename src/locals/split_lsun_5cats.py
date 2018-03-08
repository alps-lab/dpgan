#!/usr/bin/env python
import argparse
import random
import glob
import os


def sample(config):
    random_obj = random.Random(config.seed)
    for subdir in os.listdir(config.source_dir):
        path = os.path.join(config.source_dir, subdir)
        names = sorted(glob.glob(os.path.join(path, "*.jpg")))

        random_obj.shuffle(names)
        public_size = int(len(names) * config.ratio)
        public = names[:public_size]
        private = names[public_size:]

        private_dir = os.path.join(
            config.private_dir, subdir)
        public_dir = os.path.join(
            config.public_dir, subdir
        )
        os.makedirs(private_dir, exist_ok=True)
        os.makedirs(public_dir, exist_ok=True)

        for name in public:
            basename = os.path.basename(name)
            os.link(name, os.path.join(public_dir, basename))

        for name in private:
            basename = os.path.basename(name)
            os.link(name, os.path.join(private_dir, basename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", metavar="SOURCEDIR")
    parser.add_argument("private_dir", metavar="PRIVATEDIR")
    parser.add_argument("public_dir", metavar="PUBLICDIR")
    parser.add_argument("-r", "--ratio", dest="ratio", type=float, default=0.02)
    parser.add_argument("-s", "--seed", type=int, default=2048)

    sample(parser.parse_args())