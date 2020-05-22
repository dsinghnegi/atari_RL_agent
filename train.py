import os
import argparse

from tqdm import tqdm
import torch

from models.dqn import DQNAgent
from preprocessing import BreakoutNoFrameskip as BNF
from utils.replay_buffer import ReplayBuffer
from utils import utils
from utils.helper import play_and_record, compute_td_loss, evaluate

ENV_LIST=['BreakoutNoFrameskip-v4']

def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-e", "--environment", default="BreakoutNoFrameskip-v4" ,help="envirement to play")
	opt = ap.parse_args()
	return opt


def train(env,agent,target_network,device):
	state = env.reset()


	exp_replay = ReplayBuffer(10**4)
	for i in range(100):
		if not utils.is_enough_ram(min_available_gb=0.1):
			print("""
				Less than 100 Mb RAM available. 
				Make sure the buffer size in not too huge.
				Also check, maybe other processes consume RAM heavily.
				"""
				 )
			break
		play_and_record(state, agent, env, exp_replay, n_steps=10**2)
		if len(exp_replay) == 10**4:
			break


	print("Experience Reply buffer : {}".format(len(exp_replay)))

	timesteps_per_epoch = 1
	batch_size = 16
	total_steps = 3 * 10**6
	decay_steps = 10**6

	opt = torch.optim.Adam(agent.parameters(), lr=1e-4)

	init_epsilon = 1
	final_epsilon = 0.1

	loss_freq = 50
	refresh_target_network_freq = 5000
	eval_freq = 5000

	max_grad_norm = 50

	n_lives = 5


	mean_rw_history = []
	td_loss_history = []
	grad_norm_history = []
	initial_state_v_history = []
	step = 0

	_, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)
	# td_loss_history.append(loss_t)
	# train
	states, actions, rewards, next_states, is_done=exp_replay.sample(batch_size)

	loss = compute_td_loss(states, actions, rewards, next_states, is_done,
			agent, target_network,
			gamma=0.99,
			check_shapes=False,
			device=device)


	score=evaluate(make_env(clip_rewards=True, seed=444), agent, n_games=3 * n_lives, greedy=True)
	print("Score without training: {}".format(score))

	

	state = env.reset()
	for step in trange(step, total_steps + 1):
		if not utils.is_enough_ram():
			print('less that 100 Mb RAM available, freezing')
			print('make sure everythin is ok and make KeyboardInterrupt to continue')
			try:
				while True:
					pass
			except KeyboardInterrupt:
				pass

		agent.epsilon = utils.linear_decay(init_epsilon, final_epsilon, step, decay_steps)

		# play
		_, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

		# train
		states_bn, actions_bn, rewards_bn, next_states_bn, is_done_bn=exp_replay.sample(batch_size)

		loss = compute_td_loss(states_bn, actions_bn, rewards_bn, next_states_bn, is_done_bn,
						agent, target_network,
						gamma=0.99,
						check_shapes=False,
						device=device)

		loss.backward()
		grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
		opt.step()
		opt.zero_grad()

		if step % loss_freq == 0:
			td_loss_history.append(loss.data.cpu().item())
			grad_norm_history.append(grad_norm)

		if step % refresh_target_network_freq == 0:
			target_network.load_state_dict(agent.state_dict())

		if step % eval_freq == 0:
			mean_rw_history.append(evaluate(
				make_env(clip_rewards=True, seed=step), agent, n_games=3 * n_lives, greedy=True)
			)
			initial_state_q_values = agent.get_qvalues(
				[make_env(seed=step).reset()]
			)
			initial_state_v_history.append(np.max(initial_state_q_values))

			clear_output(True)
			print("buffer size = %i, epsilon = %.5f" %
				  (len(exp_replay), agent.epsilon))

			plt.figure(figsize=[16, 9])

			plt.subplot(2, 2, 1)
			plt.title("Mean reward per life")
			plt.plot(mean_rw_history)
			plt.grid()

			assert not np.isnan(td_loss_history[-1])
			plt.subplot(2, 2, 2)
			plt.title("TD loss history (smoothened)")
			plt.plot(utils.smoothen(td_loss_history))
			plt.grid()

			plt.subplot(2, 2, 3)
			plt.title("Initial state V")
			plt.plot(initial_state_v_history)
			plt.grid()

			plt.subplot(2, 2, 4)
			plt.title("Grad norm history (smoothened)")
			plt.plot(utils.smoothen(grad_norm_history))
			plt.grid()

			plt.show()




def main():
	opt=get_args()	

	assert opt.environment in ENV_LIST, \
		"Unsupported environment: {} \nSupported environemt: {}".format(opt.environment, ENV_LIST)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	env = BNF.make_env()
	state_shape = env.observation_space.shape
	n_actions = env.action_space.n
		

	agent = DQNAgent(state_shape, n_actions, epsilon=1).to(device)
	target_network = DQNAgent(state_shape, n_actions).to(device)

	train(env,agent,target_network,device)
	


	

		





if __name__ == '__main__':
	main()