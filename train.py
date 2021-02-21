import argparse
import os
import re

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import environment
import tpu
from models import DQNAgent
from utils import utils
from utils.helper import play_and_record, compute_td_loss, evaluate
from utils.replay_buffer import ReplayBuffer


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--environment", default="BreakoutDeterministic-v4", help="Envirement to play")
    ap.add_argument("-l", "--log_dir", default="logs", help="Logs dir for tensorboard")
    ap.add_argument("-t", "--train_dir", default="train_dir", help="Checkpoint directory")
    ap.add_argument("-c", "--checkpoint", default=None, help="Checkpoint for agent")
    ap.add_argument("--double_dqn", action='store_true', help="Enable double_dqn")
    ap.add_argument("--dueling", action='store_true', help="Enable dueling dqn")
    ap.add_argument("--priority_replay", action='store_true', help="Enable priority replay")
    ap.add_argument("--replay_size", default=int(10e5), type=int, help="Replay buffer size")
    ap.add_argument("--tpu", action='store_true', help="Enable TPU")
    ap.add_argument("--batch_size", default=32, type=int, help="Batch size for training")
    ap.add_argument("--init_epsilon", default=1.0, type=float, help="Intial value of epsilon")
    ap.add_argument("--final_epsilon", default=0.01, type=float, help="Final value of epsilon")
    ap.add_argument("--lr", default=1e-5, type=float, help="Learning Rate")
    ap.add_argument("--max_grad_norm", default=20.0, type=float, help="Gradient clipping")
    ap.add_argument("--gamma", default=0.99, type=float, help="Discounting factor")
    ap.add_argument("--steps", default=int(10e6), type=int, help="Training steps")
    ap.add_argument("--loss_freq", default=50, type=int, help="loss frequency")
    ap.add_argument("--target_freq", default=2500, type=int, help="Target network update frequency")
    ap.add_argument("--eval_freq", default=2500, type=int, help="Evalualtion frequency")

    opt = ap.parse_args()
    return opt


def train(make_env, agent, target_network, device, writer, opt):
    timesteps_per_epoch = 4
    batch_size = opt.batch_size
    total_steps = opt.steps
    checkpoint_path = opt.train_dir
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    init_epsilon = opt.init_epsilon
    final_epsilon = opt.final_epsilon

    loss_freq = opt.loss_freq
    refresh_target_network_freq = opt.target_freq
    eval_freq = opt.eval_freq

    max_grad_norm = opt.max_grad_norm

    priority_replay = opt.priority_replay
    replay_size = opt.replay_size
    gamma = opt.gamma

    optim = torch.optim.Adam(agent.parameters(), lr=opt.lr)

    env = make_env(clip_rewards=True)
    step = 0
    state = env.reset()
    if opt.checkpoint:
        agent.load_state_dict(torch.load(opt.checkpoint))
        target_network.load_state_dict(torch.load(opt.checkpoint))
        step = int(re.findall(r'\d+', opt.checkpoint)[-1])

    agent.epsilon = utils.linear_decay(init_epsilon, final_epsilon, step, total_steps)

    play_steps = int(10e2)
    exp_replay = ReplayBuffer(replay_size, priority_replay)
    for i in tqdm(range(100)):
        if not utils.is_enough_ram(min_available_gb=0.1):
            print("""
				Less than 100 Mb RAM available. 
				Make sure the buffer size in not too huge.
				Also check, maybe other processes consume RAM heavily.
				"""
                  )
            break
        play_and_record(state, agent, env, exp_replay, n_steps=play_steps)

    print("Experience Reply buffer : {}".format(len(exp_replay)))
    double_dqn = opt.double_dqn

    if double_dqn:
        print("Double DQN will be used for loss")

    if priority_replay:
        print("Priority replay")

    score = evaluate(make_env(clip_rewards=False), agent, greedy=True)
    print("Score without training: {}".format(score))

    env.reset()

    for step in trange(step, total_steps + 1):
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

        states_bn, actions_bn, rewards_bn, next_states_bn, is_done_bn, is_weight = exp_replay.sample(batch_size)
        optim.zero_grad()

        loss, error = compute_td_loss(states_bn, actions_bn, rewards_bn, next_states_bn, is_done_bn,
                                      agent, target_network, is_weight,
                                      gamma=gamma,
                                      check_shapes=False,
                                      device=device,
                                      double_dqn=double_dqn)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optim.step()
        exp_replay.update_priority(error)

        if step % loss_freq == 0:
            td_loss = loss.data.cpu().item()
            grad_norm = grad_norm

            assert not np.isnan(td_loss)
            writer.add_scalar("Training/TD loss history", td_loss, step)
            writer.add_scalar("Training/Grad norm history", grad_norm, step)

        if step % refresh_target_network_freq == 0:
            target_network.load_state_dict(agent.state_dict())
            torch.save(agent.state_dict(), os.path.join(checkpoint_path, "agent_{}.pth".format(step)))

        if step % eval_freq == 0:
            mean_rw = evaluate(make_env(clip_rewards=False), agent, greedy=True)

            initial_state_q_values = agent.get_qvalues(
                [make_env(seed=step).reset()]
            )
            initial_state_v = np.max(initial_state_q_values)

            print("buffer size = %i, epsilon = %.5f, mean_rw=%.2f, initial_state_v= %.2f" % (
            len(exp_replay), agent.epsilon, mean_rw, initial_state_v))

            writer.add_scalar("Eval/Mean reward per life", mean_rw, step)
            writer.add_scalar("Eval/Initial state V", initial_state_v, step)
            writer.close()

    torch.save(agent.state_dict(), os.path.join(checkpoint_path, "agent_{}.pth".format(total_steps)))


def main():
    opt = get_args()

    assert opt.environment in environment.ENV_DICT.keys(), \
        "Unsupported environment: {} \nSupported environemts: {}".format(opt.environment, environment.ENV_DICT.keys())

    writer = SummaryWriter(opt.log_dir)

    if opt.tpu:
        device = tpu.get_TPU()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ENV = environment.ENV_DICT[opt.environment]

    env = ENV.make_env()
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = DQNAgent(dueling=opt.dueling, state_shape=state_shape, n_actions=n_actions).to(device)
    target_network = DQNAgent(dueling=opt.dueling, state_shape=state_shape, n_actions=n_actions).to(device)

    writer.add_graph(agent, torch.tensor([env.reset()]).to(device))
    writer.close()

    train(ENV.make_env, agent, target_network, device, writer, opt)


if __name__ == '__main__':
    main()
