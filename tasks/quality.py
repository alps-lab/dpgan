#!/usr/bin/env python
import numpy as np
from tasks.inception_score import get_inception_score


def measure_quality(generator):
    scores = []
    for images, _ in generator:
        images = np.asarray(127.5 * (images + 1.0), np.float32)
        images = list(images)

        scores.append(get_inception_score(images, splits=1))

    return scores