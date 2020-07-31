import torch
import torch.nn as nn
import  numpy as np


class A3C(nn.Module):
	def __init__(self, n_actions, hidden=128):

		super().__init__()
		self.n_actions = n_actions
		self.hidden=hidden

		self.network=nn.Sequential(
			nn.Conv2d(4,32,3,2),
			nn.ReLU(),
			# nn.BatchNorm2d(32),

			nn.Conv2d(32,32,3,2),
			nn.ReLU(),
			# nn.BatchNorm2d(64),

			nn.Conv2d(32,32,3,2),
			nn.ReLU(),
			# nn.BatchNorm2d(64),


			nn.Flatten(),
			nn.Linear(512,self.hidden),

		)


		self.logits_network = nn.Linear(self.hidden,self.n_actions)
		self.state_value_network = nn.Linear(self.hidden,1)


		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			if isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
		   
	def forward(self, state_t):
		model_device = next(self.parameters()).device
		state_t = torch.tensor(state_t, device=model_device, dtype=torch.float)/255

		x= self.network(state_t)
		logits=self.logits_network(x)
		state_values=self.state_value_network(x)
		state_values = torch.squeeze(state_values,axis=1)

		return logits, state_values


	def sample_actions(self, agent_outputs):
		"""pick actions given numeric agent outputs (np arrays)"""
		logits, state_values = agent_outputs
		logits=logits.detach().cpu().numpy()
		state_values=state_values.detach().cpu().numpy()
		policy = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

		return np.array([np.random.choice(len(p), p=p) for p in policy])



class EnvBatch:
	def __init__(self, make_env, n_envs = 10):
		""" Creates n_envs environments and babysits them for ya' """
		self.envs = [make_env() for _ in range(n_envs)]
		
	def reset(self):
		""" Reset all games and return [n_envs, *obs_shape] observations """
		return np.array([env.reset() for env in self.envs])
	
	def step(self, actions):
		"""
		Send a vector[batch_size] of actions into respective environments
		:returns: observations[n_envs, *obs_shape], rewards[n_envs], done[n_envs,], info[n_envs]
		"""
		results = [env.step(a) for env, a in zip(self.envs, actions)]
		new_obs, rewards, done, infos = map(np.array, zip(*results))
		
		# reset environments automatically
		for i in range(len(self.envs)):
			if done[i]:
				new_obs[i] = self.envs[i].reset()
		
		return new_obs, rewards, done, infos