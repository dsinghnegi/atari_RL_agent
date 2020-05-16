from gym.core import ObservationWrapper
from gym.spaces import Box
import gym

import cv2
import matplotlib.pyplot as plt


class PreprocessAtariObs(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.img_size = (1, 64, 64)
        self.observation_space = Box(0.0, 1.0, self.img_size)


    def _to_gray_scale(self, rgb, channel_weights=[0.8, 0.1, 0.1]):
        gray=rgb[:,:,0]*channel_weights[0]+rgb[:,:,1]*channel_weights[1]+rgb[:,:,2]*channel_weights[2]
        return gray 


    def observation(self, img):
        """what happens to each observation"""

        # Here's what you need to do:
        #  * crop image, remove irrelevant parts
        #  * resize image to self.img_size
        #     (use imresize from any library you want,
        #      e.g. opencv, skimage, PIL, keras)
        #  * cast image to grayscale
        #  * convert image pixels to (0,1) range, float32 type
        img=img[30:-15,:]
        img=cv2.resize(img,(64,64))
        
        img=img.astype('float32')/255.0
        img=self._to_gray_scale(img)


        return img.reshape(-1,64,64)

if __name__ == '__main__':

    ENV_NAME = "BreakoutNoFrameskip-v4"
    env = gym.make(ENV_NAME)  # create raw env
    env = PreprocessAtariObs(env)
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n
    env.reset()
    obs, _, _, _ = env.step(env.action_space.sample())

    n_cols = 5
    n_rows = 2
    fig = plt.figure(figsize=(16, 9))
    obs = env.reset()
    for row in range(n_rows):
        for col in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
            ax.imshow(obs[0, :, :], interpolation='none', cmap='gray')
            obs, _, _, _ = env.step(env.action_space.sample())
    plt.show()
