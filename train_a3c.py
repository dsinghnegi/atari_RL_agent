import os
import argparse
import re
import concurrent.futures

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm,trange
import numpy as np
import matplotlib.pyplot as plt

from models import A3C, EnvBatch
from utils import utils
from utils.helper import evaluate_A3C,evaluate_A3C_lstm
from utils.loss import compute_A3C_loss
import environment 
import tpu


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-e", "--environment", default="BreakoutDeterministic-v4", help="Envirement to play")
	ap.add_argument("-l", "--log_dir", default="logs", help="Logs dir for tensorboard")
	ap.add_argument("-t", "--train_dir", default="train_dir", help="Checkpoint directory")
	ap.add_argument("-c", "--checkpoint", default=None, help="Checkpoint for agent")
	ap.add_argument("--tpu", action='store_true', help="Enable TPU")
	ap.add_argument("--lr", default=1e-5, type=float, help="Learning Rate")
	ap.add_argument("--max_grad_norm", default=20.0,type=float, help="Gradient clipping")
	ap.add_argument("--gamma", default=0.99, type=float, help="Discounting factor")	
	ap.add_argument("--steps", default=int(10e5), type=int, help="Training steps")
	ap.add_argument("--loss_freq", default=50, type=int, help="loss frequency")
	ap.add_argument("--eval_freq", default=2500, type=int, help="Evalualtion frequency")
	ap.add_argument("--lstm", action='store_true', help="Enable LSTM")

	
	opt = ap.parse_args()
	return opt



def train(make_env, EnvBatch, agent, device, writer, opt):
	total_steps = opt.steps
	checkpoint_path=opt.train_dir
	if not os.path.exists(checkpoint_path):
		os.mkdir(checkpoint_path)

	
	loss_freq = opt.loss_freq
	eval_freq = opt.eval_freq

	max_grad_norm = opt.max_grad_norm

	gamma=opt.gamma	

	optim = torch.optim.Adam(agent.parameters(), lr=opt.lr)

	env = make_env(lstm=opt.lstm, clip_rewards=True)
	step = 0
	state = env.reset()
	if opt.checkpoint:
		agent.load_state_dict(torch.load(opt.checkpoint))
		step=int(re.findall(r'\d+', opt.checkpoint)[-1])

	evaluate= evaluate_A3C_lstm if opt.lstm else evaluate_A3C	

	score=evaluate(make_env(lstm=opt.lstm, clip_rewards=False), agent)
	print("Score without training: {}".format(score))


	env_batch = EnvBatch(make_env=make_env, n_envs=128, clip_rewards=True, lstm=opt.lstm)
	batch_states = env_batch.reset()
	grad_norm=0
	hidden_unit=None
	for step in trange(step,total_steps + 1):
		optim.zero_grad()
		with torch.no_grad():
			if opt.lstm:
				agent_outputs, hidden_unit = agent(batch_states, hidden_unit)
			else:
				agent_outputs = agent(batch_states)

		batch_actions = agent.sample_actions(agent_outputs)
		batch_next_states, batch_rewards, batch_done, _ = env_batch.step(batch_actions)
		# print(batch_rewards)
		# batch_rewards=batch_rewards.astype(np.float32)*0.01
	
		loss, entropy = compute_A3C_loss(batch_states, batch_actions, batch_rewards, 
											batch_next_states, batch_done, agent,
											hidden_unit=hidden_unit,
											gamma=gamma,device=device)
		if loss >0:
			loss.backward()
			grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
			optim.step()

		batch_states = batch_next_states

		if step % loss_freq == 0:
			td_loss=loss.data.cpu().item()
			
			assert not np.isnan(td_loss)
			writer.add_scalar("Training/loss history",td_loss,step)
			writer.add_scalar("Training/Grad norm history",grad_norm,step)
			# print("#################",entropy)
			writer.add_scalar("Policy entropy", entropy.mean(), step)
			

		if step % eval_freq == 0:
			mean_rw=evaluate(make_env(clip_rewards=False, lstm=opt.lstm), agent)
			print("MEAN REWARD:{}".format(mean_rw))
			writer.add_scalar("Mean reward", mean_rw, step)
			writer.close()
			torch.save(agent.state_dict(), os.path.join(checkpoint_path,"agent_{}.pth".format(step)))


	torch.save(agent.state_dict(), os.path.join(checkpoint_path,"agent_{}.pth".format(total_steps)))



def main():
	opt=get_args()	

	assert opt.environment in environment.ENV_DICT.keys(), \
		"Unsupported environment: {} \nSupported environemts: {}".format(opt.environment, environment.ENV_DICT.keys())


	writer = SummaryWriter(opt.log_dir)
	
	if opt.tpu:
		device =tpu.get_TPU()
	else:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	ENV=environment.ENV_DICT[opt.environment]

	env = ENV.make_env(lstm=opt.lstm)
	state_shape = env.observation_space.shape
	n_actions = env.action_space.n
		

	agent = A3C(n_actions=n_actions, lstm=opt.lstm).to(device)
	
	writer.add_graph(agent, torch.tensor([env.reset()]).to(device))
	writer.close()

	train(ENV.make_env, EnvBatch, agent, device, writer, opt)
	




if __name__ == '__main__':
	main()