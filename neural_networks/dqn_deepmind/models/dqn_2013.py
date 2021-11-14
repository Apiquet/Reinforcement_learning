#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deep Q-learning model from: https://arxiv.org/pdf/1312.5602.pdf
"""

from tensorflow.keras.layers import Conv2D, Dense, Flatten


class DQN(nn.Module):
    def __init__(self, n_input_channels, n_actions):
        super(DQN, self).__init__()

        self.model = nn.Sequential(
            Conv2D(filters=16,
                   kernel_size=(8, 8),
                   activation="relu",
                   stride=4,
                   name="Conv1"),
            Conv2D(filters=32,
                   kernel_size=(4, 4),
                   activation="relu",
                   stride=2,
                   name="Conv2"),
            Flatten(),
            Dense(256, activation="relu", name="Dense1")
            Dense(n_actions, name="Output")
        )

    def call(self, x):
        return self.model(x)
