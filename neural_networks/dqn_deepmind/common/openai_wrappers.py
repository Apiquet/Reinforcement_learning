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


class ScaledFrom0To1Wrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class Res84x84x1Wrapper(gym.ObservationWrapper):
    """Change input resolution from (210, 160, 3) to (84, 84, 1)
    Convert to grayscale with ponderation: r*0.299, g*0.587, b*0.114
    """
    def __init__(self, env=None, skip=4):
        super(Res84x84x1Wrapper, self).__init__(env)

    def observation(self, obs):
        frame = cv2.resize(
            obs, (110, 84), interpolation=cv2.INTER_AREA).astype(
                np.float32)
        frame = frame[:, 13:110-13]
        frame = frame[:, :, 0] * 0.299 + frame[:, :, 1] * 0.587 + \
            frame[:, :, 2] * 0.114
        assert np.prod(frame.shape) == 84*84*1
        return frame

class StackLast4Wrapper(gym.ObservationWrapper):
    """Stack last 4 frames as input"""
    def __init__(self, env):
        super(StackLast4Wrapper, self).__init__(env)

    def reset(self):
        self.buffer = np.zeros(
            (84, 84, 4), dtype=np.float32)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:, :, :-1] = self.buffer[:, :, 1:]
        self.buffer[:, :, -1] = observation
        return self.buffer

class Skip4FramesAndReturnMaxFrom2FramesWrapper(gym.Wrapper):
    def __init__(self, env):
        """Repeat action for 4 frames, get max from last 2 to avoid blink"""
        super(Skip4FramesAndReturnMaxFrom2FramesWrapper, self).__init__(env)
        self.frames_buffer = collections.deque(maxlen=2)

    def step(self, action):
        sum_reward = 0.0
        for _ in range(4):
            obs, reward, done, info = self.env.step(action)
            self.frames_buffer.append(obs)
            sum_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self.frames_buffer), axis=0)
        return max_frame, sum_reward, done, info

    def reset(self):
        self.frames_buffer.clear()
        obs = self.env.reset()
        self.frames_buffer.append(obs)
        return obs
