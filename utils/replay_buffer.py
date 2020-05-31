# This code is modified on 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = {"obses_t":[], "actions":[], "rewards":[], "obses_tp1":[], "dones":[]}
        self._maxsize = size
        self._next_idx = 0
        self._probabilities =0

    def __len__(self):
        return len(self._storage["obses_t"])

    def add(self, obs_t, action, reward, obs_tp1, done):
        data={  "obses_t":np.array([obs_t]),
                "actions":np.array([action]),
                "rewards":np.array([reward]),
                "obses_tp1":np.array([obs_tp1]),
                "dones":np.array([done])
            }
        if len(self)==0:
            self._storage=data

        if self._next_idx >= len(self):
            for k in data.keys():
                self._storage[k]=np.vstack((self._storage[k],data[k]))
        else:
            for k in data.keys():
                self._storage[k][self._next_idx] = data[k]
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        return (
            self._storage["obses_t"][idxes],
            self._storage["actions"][idxes],
            self._storage["rewards"][idxes],
            self._storage["obses_tp1"][idxes],
            self._storage["dones"][idxes],
        )

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [
            random.randint(0, len(self._storage) - 1)
            for _ in range(batch_size)
        ]
        return self._encode_sample(idxes)