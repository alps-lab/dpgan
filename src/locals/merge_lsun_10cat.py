#!/usr/bin/env python
import os
import argparse
import glob


def main(config):
    os.makedirs(config.output_dir, exist_ok=True)
    for subdir in os.listdir(config.public_dir):
        path = os.path.join(config.public_dir, subdir)
        if os.path.isdir(path):
            os.makedirs(os.path.join(config.output_dir, subdir), exist_ok=True)
            for name in glob.glob(os.path.join(path, "*.jpg")):
                basename = os.path.basename(name)
                os.link(name, os.path.join(config.output_dir, subdir, basename))

    for subdir in os.listdir(config.private_dir):
        path = os.path.join(config.private_dir, subdir)
        if os.path.isdir(path):
            os.makedirs(os.path.join(config.output_dir, subdir), exist_ok=True)
            for name in glob.glob(os.path.join(path, "*.jpg")):
                basename = os.path.basename(name)
                os.link(name, os.path.join(config.output_dir, subdir, basename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("public_dir", metavar="PUBLICDIR")
    parser.add_argument("private_dir", metavar="PRIVATEDIR")
    parser.add_argument("output_dir", metavar="OUTPUTDIR")

    main(parser.parse_args())
