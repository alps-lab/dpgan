#!/usr/bin/env python
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", metavar="OUTPUT_DIR")
    parser.add_argument("input_dirs", metavar="INPUT_DIRS", nargs="+")

    config = parser.parse_args()
    os.makedirs(config.output_dir, exist_ok=True)

    all_fullnames = []
    for input_dir in config.input_dirs:
        names = [name for name in os.listdir(input_dir)
                 if name.endswith(".png")]
        fullnames = [os.path.join(input_dir, name) for name in names]
        all_fullnames.extend(fullnames)

    for i, fullname in enumerate(all_fullnames):
        os.link(fullname, os.path.join(config.output_dir, "%d.png" % i))

