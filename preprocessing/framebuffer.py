import numpy as np
from gym.spaces.box import Box
from gym.core import Wrapper


class FrameBuffer(Wrapper):
    def __init__(self, env, n_frames=4):
        """A gym wrapper that reshapes, crops and scales image into the desired shapes"""
        super(FrameBuffer, self).__init__(env)
        n_channels, height, width = env.observation_space.shape
        obs_shape = [n_channels * n_frames, height, width]
        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.framebuffer = np.zeros(obs_shape, 'uint8')

    def reset(self):
        """resets breakout, returns initial frames"""
        self.framebuffer = None
        self.update_buffer(self.env.reset())
        return self.framebuffer

    def step(self, action):
        """plays breakout for 1 step, returns frame buffer"""
        new_img, reward, done, info = self.env.step(action)
        self.update_buffer(new_img)

        return self.framebuffer, reward, done, info

    def update_buffer(self, img):
        if self.framebuffer is None:
            self.framebuffer=np.repeat(img, 4, axis=0)
        self.framebuffer = np.append(self.framebuffer[1:, :, :], img, axis=0)
      