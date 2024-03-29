import cv2
import gym
from gym.core import ObservationWrapper
from gym.spaces import Box

from preprocessing import atari_wrappers
from preprocessing.framebuffer import FrameBuffer

ENV_NAME = 'BreakoutNoFrameskip-v4'


class PreprocessAtariObs(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.img_size = (1, 64, 64)
        self.observation_space = Box(0.0, 1.0, self.img_size)

    def _to_gray_scale(self, rgb, channel_weights=[0.5, 0.2, 0.3]):
        gray = rgb[:, :, 0] * channel_weights[0] + rgb[:, :, 1] * channel_weights[1] + rgb[:, :, 2] * channel_weights[2]
        return gray

    def observation(self, img):
        """what happens to each observation"""

        img = img[32:-15, 7:-7]
        img = cv2.resize(img, (64, 64))

        img = cv2.split(img)[0]
        img = cv2.equalizeHist(img)
        return img.reshape(self.img_size)


def PrimaryAtariWrap(env, clip_rewards=True):
    assert 'NoFrameskip' in env.spec.id

    # This wrapper holds the same action for <skip> frames and outputs
    # the maximal pixel value of 2 last frames (to handle blinking
    # in some envs)
    env = atari_wrappers.MaxAndSkipEnv(env, skip=4)

    # This wrapper sends done=True when each life is lost
    # (not all the 5 lives that are givern by the game rules).
    # It should make easier for the agent to understand that losing is bad.
    env = atari_wrappers.EpisodicLifeEnv(env)

    # This wrapper laucnhes the ball when an episode starts.
    # Without it the agent has to learn this action, too.
    # Actually it can but learning would take longer.
    env = atari_wrappers.FireResetEnv(env)

    # This wrapper transforms rewards to {-1, 0, 1} according to their sign
    if clip_rewards:
        env = atari_wrappers.ClipRewardEnv(env)

    # This wrapper is yours :)
    env = PreprocessAtariObs(env)
    return env


def make_env(clip_rewards=True, seed=None):
    env = gym.make(ENV_NAME)  # create raw env
    if seed is not None:
        env.seed(seed)
    env = PrimaryAtariWrap(env, clip_rewards)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env
