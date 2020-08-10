import os
import argparse
import re

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

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
	ap.add_argument("--num_processes", default=4,type=int, help="Number of parallel environments")
	ap.add_argument("--gamma", default=0.99, type=float, help="Discounting factor")	
	ap.add_argument("--total_steps", default=int(10e5), type=int, help="Training steps")
	ap.add_argument("--num_steps", default=int(5), type=int, help="number steps for update")
	ap.add_argument("--loss_freq", default=50, type=int, help="loss frequency")
	ap.add_argument("--eval_freq", default=2500, type=int, help="Evalualtion frequency")
	ap.add_argument("--lstm", action='store_true', help="Enable LSTM")

	
	opt = ap.parse_args()
	return opt

# Sharing the parameters
# https://github.com/ikostrikov/pytorch-a3c/blob/master/main.py
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(make_env, agent, shared_agent, optim, device, writer, opt, process_number):
	total_steps = opt.total_steps
	num_steps = opt.num_steps
	checkpoint_path=opt.train_dir

	if process_number==0 and not os.path.exists(checkpoint_path):
		os.mkdir(checkpoint_path)
	
	loss_freq = opt.loss_freq
	eval_freq = opt.eval_freq

	max_grad_norm = opt.max_grad_norm

	gamma=opt.gamma	

	agent=agent.to(device)
	shared_agent=shared_agent.to(device)
	agent.train()

	env = make_env(lstm=opt.lstm, clip_rewards=True)
	step = 0
	state = env.reset()
	if opt.checkpoint:
		agent.load_state_dict(torch.load(opt.checkpoint))
		step=int(re.findall(r'\d+', opt.checkpoint)[-1])

	evaluate= evaluate_A3C_lstm if opt.lstm else evaluate_A3C	

	# score=evaluate(make_env(lstm=opt.lstm, clip_rewards=False), agent)
	# print("Score without training: {}".format(score))

	
	env = make_env(clip_rewards=True, lstm=opt.lstm)
	state = env.reset()
	grad_norm=0
	hidden_unit=None

	for step in range(step,total_steps + 1):
		agent.load_state_dict(shared_agent.state_dict())

		state_list= []
		next_state_list= []
		reward_list= []
		done_list= []
		action_list= []

		for _ in range(num_steps):
			if opt.lstm:
				agent_outputs, hidden_unit = agent([state], hidden_unit)
			else:
				agent_outputs = agent([state])

			action = agent.sample_actions(agent_outputs)
			next_state, reward, done, _ = env.step(action)

			action_list.append(action)
			state_list.append(state)
			next_state_list.append(next_state)
			done_list.append(done)
			reward_list.append(reward)
			
			if done:
				state= env.reset()
				break

			state=next_state

		
		loss, entropy = compute_A3C_loss(state_list, action_list, reward_list, 
											next_state_list, done_list, agent,
											hidden_unit=hidden_unit,
											gamma=gamma,device=device)

		optim.zero_grad()
		loss.backward()
		grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
		ensure_shared_grads(agent, shared_agent)
		optim.step()

		if process_number==0 and step % loss_freq == 0:
			td_loss=loss.data.cpu().item()
			
			assert not np.isnan(td_loss)
			print("{} Loss: {} process:{}".format(step, td_loss, process_number+1))
			writer.add_scalar("Loss/process:{}".format(process_number+1), td_loss, step)
			writer.add_scalar("Grad norm/process:{}".format(process_number+1), grad_norm, step)
			writer.add_scalar("Policy entropy/process:{}".format(process_number+1), entropy.mean(), step)
			

		if step % eval_freq == 0:
			if process_number==0:
				mean_rw=evaluate(make_env(clip_rewards=False, lstm=opt.lstm), agent)
				print("MEAN REWARD:{}".format(mean_rw))
				writer.add_scalar("Mean reward", mean_rw, step)
				
				torch.save(agent.state_dict(), os.path.join(checkpoint_path,"agent_{}.pth".format(step)))
				
				writer.close()
			
	if process_number==0:
		torch.save(agent.state_dict(), os.path.join(checkpoint_path,"agent_{}.pth".format(total_steps)))



def main():
	opt=get_args()	

	assert opt.environment in environment.ENV_DICT.keys(), \
		"Unsupported environment: {} \nSupported environemts: {}".format(opt.environment, environment.ENV_DICT.keys())


	writer = SummaryWriter(opt.log_dir)
	
	if opt.tpu:
		device =tpu.get_TPU()
	else:
		device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

	ENV=environment.ENV_DICT[opt.environment]

	env = ENV.make_env(lstm=opt.lstm)
	state_shape = env.observation_space.shape
	n_actions = env.action_space.n
		

	shared_agent = A3C(n_actions=n_actions, lstm=opt.lstm)
	
	# NOTE: this is required for the ``fork`` method to work
	# For more details: https://pytorch.org/docs/stable/notes/multiprocessing.html
	shared_agent.share_memory()
	optim = torch.optim.Adam(shared_agent.parameters(), lr=opt.lr)

	# writer.add_graph(shared_agent, torch.tensor([env.reset()]).to(device))
	# writer.close()


	processes = []

	for rank in range(0,opt.num_processes):
		p = mp.Process(target=train, args=(ENV.make_env, A3C(n_actions=n_actions, lstm=opt.lstm), shared_agent, optim, device, writer, opt, rank))
		p.start()
		processes.append(p)

	for p in processes:
		p.join()


if __name__ == '__main__':
	main()