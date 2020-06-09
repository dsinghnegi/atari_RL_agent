import torch
import numpy as np

def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _ = env.step(action)
            reward += r
 
            if done:
                break

        rewards.append(reward)
    return np.mean(rewards)




def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer. 
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0
    # env.reset()

    # Play the game for n_steps as per instructions above
    for _ in range(n_steps):
      qvalues = agent.get_qvalues([s])
      action=agent.sample_actions(qvalues)[0]
      
      s_n, r, done, _ = env.step(action)
      r= -1.0 if done else r
      exp_replay.add(s, action, r, s_n, done)
      sum_rewards+=r
      s=s_n
      if done:
        s=env.reset()
          
    return sum_rewards, s


def compute_td_loss(states, actions, rewards, next_states, is_done,
                    agent, target_network,is_weight,
                    gamma=0.99,
                    check_shapes=False,
                    device=torch.device('cpu'),double_dqn=True):
    
  
    """ Compute td loss using torch operations only. Use the formulae above. """
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
    is_not_done = 1 - is_done
    
    # get q-values for all actions in current states

    predicted_qvalues = agent(states)

     # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(
            len(actions)), actions]


    # compute q-values for all actions in next states
    predicted_next_qvalues = target_network(next_states)
    
    
    if double_dqn:
        next_actions=agent(next_states).argmax(axis=-1)
    else:
        next_actions=target_network(next_states).argmax(axis=-1)

    # compute V*(next_states) using predicted next q-values
    next_state_values=predicted_next_qvalues[range(len(next_actions)), next_actions]

    assert next_state_values.dim(
    ) == 1 and next_state_values.shape[0] == states.shape[0], "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    # print(predicted_qvalues.shape,rewards.shape)
    target_qvalues_for_actions = rewards+is_not_done*(gamma*next_state_values)

    error=torch.abs(predicted_qvalues_for_actions -
                       target_qvalues_for_actions.detach())
    
    loss =torch.mean(torch.from_numpy(is_weight).to(device).detach()
                * torch.pow(predicted_qvalues_for_actions - target_qvalues_for_actions.detach(),2))


    if check_shapes:
        assert predicted_next_qvalues.data.dim(
        ) == 2, "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim(
        ) == 1, "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim(
        ) == 1, "there's something wrong with target q-values, they must be a vector"

    return loss,error.detach().cpu().numpy()