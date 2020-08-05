import torch
import torch.nn as nn
from torch.autograd import Variable
import  numpy as np


class A3C(nn.Module):
	def __init__(self, n_actions, hidden=128, lstm=False):

		super().__init__()
		assert not lstm, \
				"LSTM Agent is not invoked"
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


class A3C_lstm(nn.Module):
	def __init__(self, n_actions, hidden=512, lstm=True):

		super().__init__()
		assert lstm, \
				"LSTM Agent is invoked"
		self.n_actions = n_actions
		self.hidden=hidden

		self.network=nn.Sequential(
			nn.Conv2d(1,32,3,2),
			nn.ReLU(),
			# nn.BatchNorm2d(32),

			nn.Conv2d(32,64,3,2),
			nn.ReLU(),
			# nn.BatchNorm2d(64),

			nn.Conv2d(64,64,3,2),
			nn.ReLU(),
			# nn.BatchNorm2d(64),

			nn.Conv2d(64,64,3,2),
			nn.ReLU(),
			
			nn.Flatten(),
		)

		self.lstm_layer=nn.LSTMCell(1024, self.hidden)


		self.logits_network = nn.Sequential(
			nn.ReLU(), 
			nn.Linear(self.hidden,self.n_actions),
			)

		self.state_value_network = nn.Sequential(
			nn.ReLU(), 
			nn.Linear(self.hidden,1),
			)


		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			if isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			if isinstance(m, nn.LSTMCell):
				nn.init.kaiming_normal_(m.weight_ih, mode='fan_out', nonlinearity='relu')
				nn.init.kaiming_normal_(m.weight_hh, mode='fan_out', nonlinearity='relu')
				 
	def forward(self, state_t, hidden_unit=None):
		model_device = next(self.parameters()).device
		state_t = torch.tensor(state_t, device=model_device, dtype=torch.float)/255
		
		if hidden_unit is None:
			cx = Variable(torch.zeros(state_t.size(0), 512).cuda())
			hx = Variable(torch.zeros(state_t.size(0), 512).cuda())
		else:
			(hx, cx)=hidden_unit
		
		x= self.network(state_t)
		x = x.view(x.size(0), -1)
		hx, cx=self.lstm_layer(x,(hx, cx))
		logits=self.logits_network(hx)
		state_values=self.state_value_network(hx)
		state_values = torch.squeeze(state_values,axis=1)

		return (logits, state_values), (hx, cx)


	def sample_actions(self, agent_outputs):
		"""pick actions given numeric agent outputs (np arrays)"""
		logits, state_values = agent_outputs
		logits=logits.detach().cpu().numpy()
		state_values=state_values.detach().cpu().numpy()
		policy = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

		return np.array([np.random.choice(len(p), p=p) for p in policy])

	def best_actions(self, agent_outputs):
		"""pick actions given numeric agent outputs (np arrays)"""
		logits, state_values = agent_outputs
		logits=logits.detach().cpu().numpy()
		state_values=state_values.detach().cpu().numpy()
		policy = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

		return np.array([np.argmax(p) for p in policy])
	


class EnvBatch:
	def __init__(self, make_env, n_envs = 10, clip_rewards=True,lstm=False):
		""" Creates n_envs environments and babysits them for ya' """
		self.envs = [make_env(clip_rewards=clip_rewards, lstm=lstm) for _ in range(n_envs)]
		
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