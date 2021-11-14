#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrappers for openai gym environments, strongly inspired of
openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""

import cv2
import gym
import numpy as np
import collections


class Res84WithHeightCrop(gym.ObservationWrapper):
    """Change input resolution from (210, 160, 3) to (84, 84, 1)
    Convert to grayscale with ponderation: r*0.299, g*0.587, b*0.114
    """
    def __init__(self, env=None, skip=4):
        super(Res84WithHeightCrop, self).__init__(env)

    def observation(self, obs):
        frame = cv2.resize(
            obs, (110, 84), interpolation=cv2.INTER_AREA)
        frame = frame[:, 13:110-13]
        frame = frame[:, :, 0] * 0.299 + frame[:, :, 1] * 0.587 + \
            frame[:, :, 2] * 0.114
        assert np.prod(frame.shape) == 84*84*1
        return frame
