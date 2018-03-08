#!/usr/bin/env python
from abc import ABCMeta, abstractmethod


class Scheduler(object):

    __meta__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_critic_steps(self, step):
        pass

