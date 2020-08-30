import os
import argparse
import re

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as _mp
from torch.nn import functional as F

from tqdm import tqdm,trange
import numpy as np
import matplotlib.pyplot as plt

from optim import GlobalAdam, SharedAdam
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
	ap.add_argument("--max_grad_norm", default=50.0,type=float, help="Gradient clipping")
	ap.add_argument("--num_processes", default=4,type=int, help="Number of parallel environments")
	ap.add_argument("--value_loss_coef", default=0.5, type=float, help="value loss coef")
	ap.add_argument("--gae_lambda", default=1.0, type=float, help="gae lambda")
	ap.add_argument("--entropy_coef", default=0.01, type=float, help="entropy coef")
	ap.add_argument("--gamma", default=0.99, type=float, help="Discounting factor")	
	ap.add_argument("--total_steps", default=int(10e6), type=int, help="Training steps")
	ap.add_argument("--num_steps", default=20, type=int, help="number steps for update")
	ap.add_argument("--loss_freq", default=50, type=int, help="loss frequency")
	ap.add_argument("--eval_freq", default=2500, type=int, help="Evalualtion frequency")
	ap.add_argument("--lstm", action='store_true', help="Enable LSTM")
	ap.add_argument("--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
		help="Device for training")

	
	opt = ap.parse_args()
	return opt


def train(make_env, shared_agent, optim, device, opt, process_number):
	total_steps = opt.total_steps
	num_steps = opt.num_steps
	loss_freq = opt.loss_freq
	eval_freq = opt.eval_freq
	max_grad_norm = opt.max_grad_norm
	gamma=opt.gamma	


	checkpoint_path=opt.train_dir

	if process_number==0 and not os.path.exists(checkpoint_path):
		os.mkdir(checkpoint_path)
	

	step = 0
	evaluate= evaluate_A3C_lstm if opt.lstm else evaluate_A3C	

	
	env = make_env(clip_rewards=False, lstm=opt.lstm)
	state = env.reset()

	grad_norm=0
	hidden_unit=None

	n_actions = env.action_space.n

	agent=A3C(n_actions=n_actions, lstm=opt.lstm)
	agent=agent.to(device)
	shared_agent=shared_agent.to(device)
	agent.train()
	shared_agent.train()

	if opt.checkpoint:
		agent.load_state_dict(torch.load(opt.checkpoint, map_location=torch.device(device)))
		step=int(re.findall(r'\d+', opt.checkpoint)[-1])



	if process_number==0:
		writer = SummaryWriter(opt.log_dir)
	
	episode_length=0
	episode_reward=0

	for step in range(step,total_steps + 1):
		agent.load_state_dict(shared_agent.state_dict())
		
		log_policies = []
		values = []
		rewards = []
		entropies = []

		for _ in range(num_steps):
			(logits, value), hidden_unit = agent([state], hidden_unit)
			

			policy = F.softmax(logits, dim=1)
			log_policy = F.log_softmax(logits, dim=1)
			entropy = -(policy * log_policy).sum(1, keepdim=True)

			action = agent.sample_actions((logits, value))
			state, reward, done, _ = env.step(action.squeeze())

			
			episode_reward+=reward
			episode_length+=1
			
			if done:
				state= env.reset()
				hidden_unit=None
		
				if process_number==0:
					writer.add_scalar("Episode/Length", episode_length, step)
					writer.add_scalar("Episode/Reward",episode_reward, step)
		
				episode_length=0
				episode_reward=0

			values.append(value)
			log_policies.append(log_policy.gather(1, action))
			rewards.append(np.sign(reward))
			entropies.append(entropy)

			if done:				
				break
		
		R = torch.zeros((1, 1), dtype=torch.float).to(device)
		if not done:
			(_, R), _ = agent([state], hidden_unit)
	
		gae = torch.zeros((1, 1), dtype=torch.float).to(device)
		actor_loss = 0
		critic_loss = 0
		entropy_loss = 0
		next_value = R

		for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
			gae = gae * gamma * opt.gae_lambda
			gae = gae + reward + gamma * next_value - value
			next_value = value
			
			actor_loss = actor_loss + log_policy * gae.detach()
			
			R = R * gamma + reward
			
			critic_loss = critic_loss + (R - value) ** 2 / 2
			entropy_loss = entropy_loss + entropy
		
		policy_loss=-actor_loss- opt.entropy_coef * entropy_loss
		total_loss = policy_loss + opt.value_loss_coef* critic_loss
		
		optim.zero_grad()
		
		total_loss.backward()
		grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)

		for local_param, global_param in zip(agent.parameters(), shared_agent.parameters()):
			if global_param.grad is not None:
				break
			global_param._grad = local_param.grad

		optim.step()


		if process_number==0 and step % loss_freq == 0:
			loss=total_loss.data.cpu().item()

			print("[{}] Loss: {} process: {}".format(step, loss, process_number+1))
			
			writer.add_scalar("Training/Loss", loss, step)
			writer.add_scalar("Training/Grad norm", grad_norm, step)
			writer.add_scalar("Training/Policy entropy", entropy_loss, step)
			

		if process_number==0 and step % eval_freq == 0:
				mean_rw=evaluate(make_env(clip_rewards=False, lstm=opt.lstm), agent)
				writer.add_scalar("Mean reward", mean_rw, step)
				writer.close()

				torch.save(agent.state_dict(), os.path.join(checkpoint_path,"agent_{}.pth".format(step)))
				
	if process_number==0:
		torch.save(agent.state_dict(), os.path.join(checkpoint_path,"agent_{}.pth".format(total_steps)))



def main():
	opt=get_args()	

	assert opt.environment in environment.ENV_DICT.keys(), \
		"Unsupported environment: {} \nSupported environemts: {}".format(opt.environment, environment.ENV_DICT.keys())


	
	if opt.tpu:
		device =tpu.get_TPU()
	else:
		device = opt.device

	mp = _mp.get_context("spawn")

	ENV=environment.ENV_DICT[opt.environment]

	env = ENV.make_env(lstm=opt.lstm)
	state_shape = env.observation_space.shape
	n_actions = env.action_space.n
		

	shared_agent = A3C(n_actions=n_actions, lstm=opt.lstm).to(device)
	
	shared_agent.share_memory()
	
	optim = SharedAdam(shared_agent.parameters(), lr=opt.lr)
	optim.share_memory()

	processes = []

	for rank in range(0,opt.num_processes):
		p = mp.Process(target=train, args=(ENV.make_env, shared_agent, optim, device, opt, rank))
		p.start()
		processes.append(p)

	for p in processes:
		p.join()


if __name__ == '__main__':
	main()