import random
import numpy as np
import torch

import gym
import numpy as np
import matplotlib.pyplot as plt


ENV_NAME = "BreakoutNoFrameskip-v4"


env = gym.make(ENV_NAME)
env.reset()

n_cols = 5
n_rows = 2
fig = plt.figure(figsize=(16, 9))

for row in range(n_rows):
    for col in range(n_cols):
        ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
        ax.imshow(env.render('rgb_array'))
        env.step(env.action_space.sample())
plt.show()


from gym.utils.play import play

play(env=gym.make(ENV_NAME), zoom=5, fps=30)

