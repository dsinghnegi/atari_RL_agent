import torch
from torch import nn
from torch.nn import functional as F



def compute_A3C_loss(states, actions, rewards, next_states, is_done,
					agent, gamma=0.99,	device=torch.device('cpu')):

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
	logits, state_values = agent(states)
	next_logits, next_state_values = agent(next_states)

	# There is no next state if the episode is done!
	next_state_values = next_state_values * (1 - is_done)

	# print(next_state_values)
	# probabilities and log-probabilities for all actions
	probs = F.softmax(logits,dim=1)
	# print(probs)            # [n_envs, n_actions]
	logprobs = F.log_softmax(logits,dim=1)     # [n_envs, n_actions]
	# print(logprobs)
	# log-probabilities only for agent's chosen actions
	logp_actions = torch.sum(logprobs.gather(1,actions.view(-1,1)), axis=-1) # [n_envs,]

	# print(logprobs.gather(1,actions.view(-1,1)))
	
	# Compute advantage using rewards_ph, state_values and next_state_values.
	advantage = rewards + gamma*next_state_values - state_values
	# print(advantage)
	assert len(advantage.shape) == 1, "please compute advantage for each sample, vector of shape [n_envs,]"

	# Compute policy entropy given logits_seq. Mind the "-" sign!
	entropy = -torch.sum(probs*logprobs, axis=1)
	assert len(entropy.shape) == 1, "please compute pointwise entropy vector of shape [n_envs,] "

	# Compute target state values using temporal difference formula. Use rewards_ph and next_step_values

	actor_loss = -torch.mean(logp_actions * advantage.detach()) - 0.001 * torch.mean(entropy)
	target_state_values = rewards+gamma*next_state_values
	# print(target_state_values)
	critic_loss = torch.mean((state_values - target_state_values.detach())**2)
	loss = actor_loss + critic_loss	
	
	return loss, entropy