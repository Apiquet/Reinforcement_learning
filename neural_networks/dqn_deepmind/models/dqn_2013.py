#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deep Q-learning model from: https://arxiv.org/pdf/1312.5602.pdf
"""

import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    def __init__(self, n_input_channels, n_actions):
        super(DQN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def call(self, x):
        return self.model(x)
