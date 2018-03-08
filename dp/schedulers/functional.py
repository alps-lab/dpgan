from .base import Scheduler


class FunctionalSchedular(Scheduler):

    def __init__(self, callback):
        super(Scheduler, self).__init__()
        self.callback = callback

    def get_critic_steps(self, step):
        return self.callback(step)