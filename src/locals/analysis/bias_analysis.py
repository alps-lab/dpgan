#!/usr/bin/env python
from six.moves import xrange

from collections import defaultdict
import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


PATTERN = re.compile(r"<tf.Variable '(?P<name>.*?)'.*?> (?P<mean>\S*) (?P<var>\S*)")


def read_log(path):
    record = defaultdict(list)
    with open(path) as f:
        for line in f:
            match_obj = PATTERN.match(line)
            if match_obj is not None:
                record[match_obj.group("name")].append(
                    (float(match_obj.group("mean")), float(match_obj.group("var")
                )))
    return dict(record)


def plot_it():
    record = {}
    record["after_private"] = read_log("./a.log")
    record["after_public"] = read_log("./b.log")
    record["before_private"] = read_log("./d.log")
    record["before_public"] = read_log("./c.log")

    keys = sorted(record["after_private"])
    for i, key in enumerate(keys, 1):
        plt.subplot(3, 3, i)
        plt.plot(np.linspace(0, 1, 200), [m for m, _ in record["after_private"][key][:200]],
             label="private")
        plt.plot(np.linspace(0, 1, 200), [m for m, _ in record["after_public"][key][:200]],
             label="public")
        plt.plot(np.linspace(0, 1, 200), np.ones((200,)) * np.mean([m for m, _ in record["after_private"][key][:200]]),
                 label="private_mean")
        plt.plot(np.linspace(0, 1, 200), np.ones((200,)) * np.mean([m for m, _ in record["after_public"][key][:200]]),
                 label="public_mean")
        plt.title(key, fontsize=6)
        plt.legend(fontsize=4)

    plt.savefig("bias.pdf")


if __name__ == "__main__":
    plot_it()
