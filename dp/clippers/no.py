from dp.clippers.base import Clipper


class NoClipper(Clipper):

    def __init__(self):
        super(NoClipper, self).__init__()

    def num_accountant_terms(self, step):
        return 0

    def clip_grads(self, m):
        clipped = []
        for k, v in m:
            clipped.append(v)
        return clipped

    def noise_grads(self, m, batch_size, sigma):
        return m
