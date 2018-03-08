#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
