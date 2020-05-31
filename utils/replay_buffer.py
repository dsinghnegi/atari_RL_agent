# This code is modified on 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, priority_replay=False,size,alpha=0.7,beta=0.5,eps=1e-7):
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
        self._probabilities =[]
        self.priority_replay=priority_replay
        self._eps=eps
        self.alpha=alpha
        self.beta=beta


    def __len__(self):
        return len(self._storage["obses_t"])

    def _max_priority(self):
        return np.max(self._probabilities) if self.priority_replay else 1.0

    def add(self, obs_t, action, reward, obs_tp1, done):
        data={  "obses_t":np.array([obs_t]),
                "actions":np.array([action]),
                "rewards":np.array([reward]),
                "obses_tp1":np.array([obs_tp1]),
                "dones":np.array([done])
            }

        if len(self)==0:
            self._storage=data
            self._probabilities=np.array([1.0])

        if self._next_idx >= len(self):
            self._probabilities=np.vstack(self._probabilities,[self._max_priority()])
            for k in data.keys():
                self._storage[k]=np.vstack((self._storage[k],data[k]))
        else:
            for k in data.keys():
                self._storage[k][self._next_idx] = data[k]
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def update_priority(self,td_loss):
        if self.priority_replay:
            self._probabilities[self.idxes]=np.power(torch.abs(td_loss) + self._eps,self.alpha)

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
        probabilities=self._probabilities/np.sum(self._probabilities)
        self.idxes = np.random.choice(
                  range(len(self)), 
                  batch_size,
                  p=probabilities,
                )
        if self.priority_replay:
            is_weight = np.power(len(self) * probabilities[self.idxes], -self.beta)
            is_weight /= is_weight.max()
        else:
           is_weight=np.ones(len(self.idxes)) 
        return *self._encode_sample(idxes),is_weight