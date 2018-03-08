from .base import Scheduler


class BasicScheduler(Scheduler):

    def __init__(self, num_critics):
        super(BasicScheduler, self).__init__()
        self.num_critics = num_critics

    def get_critic_steps(self, step):
        return self.num_critics
