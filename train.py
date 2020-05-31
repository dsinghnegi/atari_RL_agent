import os
import argparse
import re


from tqdm import tqdm,trange
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.dqn import DQNAgent
from preprocessing import BreakoutNoFrameskip as BNF
from utils.replay_buffer import ReplayBuffer
from utils import utils
from utils.helper import play_and_record, compute_td_loss, evaluate
from torch import nn
from torch.utils.tensorboard import SummaryWriter



ENV_LIST=['BreakoutNoFrameskip-v4']

def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-e", "--environment", default="BreakoutNoFrameskip-v4" ,help="envirement to play")
	ap.add_argument("-l", "--log_dir", default="logs" ,help="logs dir for tensorboard")
	ap.add_argument("-t", "--train_dir", default="train_dir" ,help="checkpoint directory for tensorboard")
	ap.add_argument("-c", "--checkpoint",default=None,help="checkpoint for agent")
	ap.add_argument( "--double_dqn", default=True ,help="enable double_dqn")
	ap.add_argument( "--priority_replay", default=True ,help="enable priority replay")
	
	opt = ap.parse_args()
	return opt



def train(env,make_env,agent,target_network,device,writer,checkpoint_path,opt):
	timesteps_per_epoch = 1
	batch_size = 16
	total_steps = 3 * 10**6

	optim = torch.optim.Adam(agent.parameters(), lr=1e-4)

	init_epsilon = 1
	final_epsilon = 0.1

	loss_freq = 50
	refresh_target_network_freq = 5000
	eval_freq = 5000

	max_grad_norm = 2.0

	n_lives = 5
	priority_replay=opt.priority_replay

	step = 0

	state = env.reset()
	if opt.checkpoint:
		agent.load_state_dict(torch.load(opt.checkpoint))
		target_network.load_state_dict(torch.load(opt.checkpoint))
		step=int(re.findall(r'\d+', opt.checkpoint)[-1])


	exp_replay = ReplayBuffer(10**4,priority_replay)
	for i in tqdm(range(100)):
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
	double_dqn=opt.double_dqn

	if double_dqn:
		print("Double DQN will be used for loss")




	score=evaluate(make_env(clip_rewards=True, seed=444), agent, n_games=3 * n_lives, greedy=True)
	print("Score without training: {}".format(score))

	
	loss=10**6

	state = env.reset()
	for step in trange(step,total_steps + 1):
		if not utils.is_enough_ram():
			print('less that 100 Mb RAM available, freezing')
			print('make sure everythin is ok and make KeyboardInterrupt to continue')
			try:
				while True:
					pass
			except KeyboardInterrupt:
				pass

		agent.epsilon = utils.linear_decay(init_epsilon, final_epsilon, step, total_steps)

		# play
		_, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

		# train
		states_bn, actions_bn, rewards_bn, next_states_bn, is_done_bn,is_weight=exp_replay.sample(batch_size)
		optim.zero_grad()

		loss,error = compute_td_loss(states_bn, actions_bn, rewards_bn, next_states_bn, is_done_bn,
						agent, target_network,is_weight,
						gamma=0.99,
						check_shapes=False,
						device=device,
						double_dqn=double_dqn)

		loss.backward()
		grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
		optim.step()
		exp_replay.update_priority(error)

		# print("--->",loss.data.cpu().item(),grad_norm)
		if step % loss_freq == 0:
			td_loss=loss.data.cpu().item()
			grad_norm=grad_norm

			assert not np.isnan(td_loss)
			writer.add_scalar("TD loss history (smoothened)",
                            utils.smoothen([td_loss])[-1],
                            step)
			writer.add_scalar("Grad norm history (smoothened)",
                            utils.smoothen([grad_norm])[-1],
                            step)


		if step % refresh_target_network_freq == 0:
			target_network.load_state_dict(agent.state_dict())
			torch.save(agent.state_dict(), os.path.join(checkpoint_path,"agent_{}.pth".format(step)))

		if step % eval_freq == 0:
			mean_rw=evaluate(make_env(clip_rewards=True, seed=step), agent, n_games=3 * n_lives, greedy=True)
			
			initial_state_q_values = agent.get_qvalues(
				[make_env(seed=step).reset()]
			)
			initial_state_v=np.max(initial_state_q_values)

		
			print("buffer size = %i, epsilon = %.5f, mean_rw=%.2f, initial_state_v= %.2f"   % (len(exp_replay), agent.epsilon, mean_rw, initial_state_v))

			writer.add_scalar("Mean reward per life", mean_rw, step)
			writer.add_scalar("Initial state V", initial_state_v, step)

	torch.save(agent.state_dict(), os.path.join(checkpoint_path,"agent_{}.pth".format(total_steps)))


def main():
	opt=get_args()	

	assert opt.environment in ENV_LIST, \
		"Unsupported environment: {} \nSupported environemt: {}".format(opt.environment, ENV_LIST)


	writer = SummaryWriter(opt.log_dir)
	checkpoint_path=opt.train_dir
	if not os.path.exists(checkpoint_path):
		os.mkdir(checkpoint_path)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	env = BNF.make_env()
	state_shape = env.observation_space.shape
	n_actions = env.action_space.n
		

	agent = DQNAgent(state_shape, n_actions, epsilon=1).to(device)
	target_network = DQNAgent(state_shape, n_actions).to(device)

	

	writer.add_graph(agent,torch.tensor([env.reset()]).to(device))
	writer.close()

	train(env,BNF.make_env,agent,target_network,device,writer,checkpoint_path,opt)
	




if __name__ == '__main__':
	main()