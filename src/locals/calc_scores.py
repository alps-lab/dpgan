#!/usr/bin/env python
import argparse
import numpy as np


def main(paths):
    for path in paths:
        obj = np.load(path)
        print("%s: %.2f %.2f" %
              (path, obj["scores"]["mean"], obj["scores"]["std"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_path", nargs="+", metavar="FILES")

    main(parser.parse_args().npy_path)