import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def compute_A3C_loss(states, actions, rewards, next_states, is_done,
					agent, hidden_unit=None, gamma=0.99,	device=torch.device('cpu')):

	states = torch.tensor(states, device=device, dtype=torch.float)    # shape: [batch_size, *state_shape]

	# for some torch reason should not make actions a tensor
	actions = torch.tensor(actions, device=device, dtype=torch.long)    # shape: [batch_size]
	rewards = torch.tensor(rewards, device=device, dtype=torch.float)  # shape: [batch_size]
	# shape: [batch_size, *state_shape]
	next_states = torch.tensor(next_states, device=device, dtype=torch.float)
	is_done = torch.tensor(
		is_done.astype('float32'),
		device=device,
		dtype=torch.float
	)  # shape: [batch_size]

	# logits[n_envs, n_actions] and state_values[n_envs, n_actions]
	if  hidden_unit is not None:
		(logits, state_values), hidden_unit = agent(states, hidden_unit)
		(next_logits, next_state_values),  hidden_unit = agent(next_states, hidden_unit)
	else:
		(logits, state_values) = agent(states)
		(next_logits, next_state_values) = agent(next_states)

	# There is no next state if the episode is done!
	next_state_values = next_state_values * (1 - is_done)

	for i in range(len(next_state_values)-2,-1,-1):
		next_state_values[i] = next_state_values[i+1] + gamma*rewards[i+1] 		

	probs = F.softmax(logits,dim=1)
	logprobs = F.log_softmax(logits,dim=1)     # [n_envs, n_actions]

	# log-probabilities only for agent's chosen actions
	logp_actions = torch.sum(logprobs.gather(1,actions.view(-1,1)), axis=-1) # [n_envs,]

	
	# Compute advantage using rewards_ph, state_values and next_state_values.
	advantage = rewards + gamma*next_state_values - state_values
	assert len(advantage.shape) == 1, "please compute advantage for each sample, vector of shape [n_envs,]"

	# Compute policy entropy given logits_seq. Mind the "-" sign!
	entropy = -torch.sum(probs*logprobs, axis=1)
	assert len(entropy.shape) == 1, "please compute pointwise entropy vector of shape [n_envs,] "

	# Compute target state values using temporal difference formula. Use rewards_ph and next_step_values
	actor_loss = -(logp_actions * advantage.detach()).mean() - 0.001 * (entropy).mean()
	
	# Actor or state loss
	target_state_values = rewards+gamma*next_state_values
	critic_loss = F.mse_loss(state_values, target_state_values.detach())


	loss = actor_loss + critic_loss	

	assert abs(actor_loss) < 100 and abs(critic_loss) < 100, "losses seem abnormally large"
	assert 0 <= entropy.mean() <= np.log(logprobs.shape[1]), "impossible entropy value, double-check the formula pls"
	
	return loss, entropy