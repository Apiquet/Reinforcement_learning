#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deep Q-learning model from: https://arxiv.org/pdf/1312.5602.pdf
"""

from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras import Model, Sequential


class DQN(Model):
    def __init__(self, n_input_channels, n_actions):
        super(DQN, self).__init__()

        self.model = Sequential([
            Conv2D(filters=16,
                   kernel_size=(8, 8),
                   activation="relu",
                   strides=(4, 4),
                   name="Conv1"),
            Conv2D(filters=32,
                   kernel_size=(4, 4),
                   activation="relu",
                   strides=(2, 2),
                   name="Conv2"),
            Flatten(),
            Dense(256, activation="relu", name="Dense1"),
            Dense(n_actions, name="Output")],
            name="DQN_2013"
        )
        opt = tf.keras.optimizers.Adam(lr=self.policy_learning_rate)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy')

    def call(self, x):
        return self.model(x)
