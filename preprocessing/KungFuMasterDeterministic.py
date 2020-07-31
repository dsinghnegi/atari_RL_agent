import random
import numpy as np
import gym
from gym.core import ObservationWrapper
from gym.spaces import Box
import cv2

from preprocessing.framebuffer import FrameBuffer
from preprocessing import atari_wrappers
from gym.core import Wrapper

ENV_NAME = "KungFuMasterDeterministic-v4"

class PreprocessAtariObs(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.img_size = (1, 42, 42)
        self.observation_space = Box(0.0, 1.0, self.img_size)

    def observation(self, img):
        """what happens to each observation"""

        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img=img[60:-30,5:]
        img=cv2.resize(img,(42,42),cv2.INTER_NEAREST)

        return img.reshape(-1,42,42)


def PrimaryAtariWrap(env, clip_rewards=True, scale=100):
    # This wrapper holds the same action for <skip> frames and outputs
    # the maximal pixel value of 2 last frames (to handle blinking
    # in some envs)
    env = atari_wrappers.MaxAndSkipEnv(env, skip=1)

    # This wrapper sends done=True when each life is lost
    # (not all the 5 lives that are givern by the game rules).
    # It should make easier for the agent to understand that losing is bad.
    env = atari_wrappers.EpisodicLifeEnv(env)

    # This wrapper laucnhes the ball when an episode starts.
    # Without it the agent has to learn this action, too.
    # Actually it can but learning would take longer.
    # env = atari_wrappers.FireResetEnv(env)

    # This wrapper transforms rewards to {-1, 0, 1} according to their sign
    # if clip_rewards:
    #     env = atari_wrappers.ScaleRewardEnv(env, scale=100)

    env = PreprocessAtariObs(env)
    return env


def make_env(clip_rewards=True, seed=None):
    env = gym.make(ENV_NAME)  # create raw env
    if seed is not None:
        env.seed(seed)
    env=PrimaryAtariWrap(env,clip_rewards=clip_rewards)
    env = FrameBuffer(env, n_frames=4)
    return env
