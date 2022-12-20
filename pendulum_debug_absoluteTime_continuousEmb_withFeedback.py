### Program implementing Decision Transformer on CartPole-v1 environment

## Features:
# 1. Solve RL problems using a causal transformer encoder by reformulating them as sequence prediction problem
# 2. This implementation does not use a purely offline-RL setting, as we append the new trajectories (obtained during evaluation) to the dataset
# 3. Note that input to the transformer is in (R,s,a) format and the output from the transformer will be in (s,a,R) format (since outputs are left-shifted)

## Todos / Questions:
# 1. Positional embeddings: fixed (sinusoidal) versus learnt (paper uses learnt positional embeddings)
# 2. Should we add action repeat (no loss over the repeated action - equivalent to reducing the decision making steps)
# 3. How to set K (heuristically set for now)
# 4. How to handle trajectories with len < K ? (guess: make the terminal state an absorbing state with zero reward and add a fixed action from the action space as the pad_action_token to avoid incurring loss on the action taken at the terminal state. So we can increase the lenght of the trajectory to K by appending <RT=0; sT=sT; aT=pad_action>)
# 5. Do we need a pad mask for terminal transitions when the causal mask takes care of all the masking needs?
# 6. Do we need to add exploration to actions during evaluation (and hence in the new trajectories generated) ?
# 7. Should the positional_embedding be relative to input (0 to K) or absolute (equal to episode time step)


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from copy import deepcopy
import gym
from tqdm import tqdm

from utils_transformer import *

# torch.autograd.set_detect_anomaly(True)


# class implementing embeddings (to obtain embeddings from tokens)
# note that we use linear embeddings for all 3 (states, actions, returns2go) as they can handle both discrete and continuous values (though its suboptimal for discrete values)
class Embeddings(nn.Module):
    def __init__(self, d_model, K, o_dim, a_dim, num_actions, max_return, max_timestep, dropout):
        super().__init__()
        self.state_emb = nn.Linear(o_dim, d_model, bias=False)
        self.action_emb = nn.Linear(a_dim, d_model, bias=True) # +1 for pad_action

        self.returns2go_emb = nn.Linear(1, d_model, bias=True) # +1 for pad_return2go = 0
        # self.returns2go_emb = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, d_model))

        self.timestep_emb = nn.Embedding(max_timestep+1, d_model) # precautionary +1
        # self.pos_emb = nn.Parameter(torch.randn(K, d_model)) # learnt positional embeddings
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, states, actions, returns2go, timesteps): # shape: [batch_size, K, dim]
        batch_size = states.shape[0]
        # get embeddings
        states_emb = self.state_emb(states)
        actions_emb = self.action_emb(actions)
        returns2go_emb = self.returns2go_emb(returns2go)
        timesteps_emb = self.timestep_emb(timesteps)
        # add positional embeddings
        # pos_emb = self.pos_emb
        # pos_emb = pos_emb.unsqueeze(0) # pos_emb.shape: [1, K, d_model]
        # pos_emb = pos_emb.expand(batch_size, -1, -1) # pos_emb.shape: [batch_size, K, d_model]
        states_emb = self.dropout( self.norm(states_emb + timesteps_emb) )
        actions_emb = self.dropout( self.norm(actions_emb + timesteps_emb) )
        returns2go_emb = self.dropout( self.norm(returns2go_emb + timesteps_emb) )
        # states_emb = self.dropout( self.norm(states_emb + pos_emb) )
        # actions_emb = self.dropout( self.norm(actions_emb + pos_emb) )
        # returns2go_emb = self.dropout( self.norm(returns2go_emb + pos_emb) )
        # concat returns2go, states, actions - IN THAT ORDER - into a single trajectory of length 3*K
        traj = torch.stack((returns2go_emb, states_emb, actions_emb), dim=1) # traj.shape: [batch_size, 3, K, d_model]
        traj = traj.permute(0, 2, 1, 3) # traj.shape: [batch_size, K, 3, d_model] - because for flattening into (R, s, a), we want K rows and 3 columns; not 3 rows and K columns
        traj = torch.flatten(traj, start_dim=-3, end_dim=-2) # traj.shape: [batch_size, 3K, d_model]
        return traj


# class implementing the Decision Transformer
class DecisionTransformer(nn.Module):
    def __init__(self, embeddings, encoder, d_model, num_actions, a_dim, max_action, seq_len, device):
        super().__init__()
        self.embeddings = embeddings
        self.encoder = encoder
        # self.action_proj = nn.Linear(d_model, a_dim, bias=False)
        self.action_proj = nn.Sequential( nn.Linear(d_model, a_dim, bias=False), nn.Tanh() )
        self.max_action = max_action
        self.device = device
    # function for forward prop through the transformer encoder and get action scores
    def forward(self, states, actions, returns2go, timesteps, pad_mask=None): # shape: [batch_size, K, dim]
        # convert raw data to embeddings
        traj = self.embeddings(states, actions, returns2go, timesteps) # traj.shape: [batch_size, 3K, d_model]
        # get causal mask
        batch_size, seq_len = traj.shape[0], traj.shape[1]
        causal_mask = subsequent_mask((batch_size, seq_len)).to(self.device)
        # forward prop through transformer encoder
        out = self.encoder(traj, mask_padding=pad_mask, mask_causal=causal_mask) # out.shape: [batch_size, 3K, d_model]
        # extract predicted actions scores
        K = seq_len // 3
        out = out.reshape(batch_size, K, 3, d_model)
        out = out.permute(0, 2, 1, 3) # out.shape: [batch_size, 3, K, d_model]
        out_states, out_actions, out_returns2go = out[:,0], out[:,1], out[:,2] # NOTE that output from the transformer is in (s,a,R) format; since input was (R,s,a) format
        # project action_scores to num_actions
        action_scores = self.action_proj(out_actions) # action_scores.shape: [batch_size, K, num_actions]
        action_scores = action_scores * self.max_action
        return action_scores
    # function to autoregressively predict an action
    def predict(self, states, actions, returns2go, timesteps, action_idx, pad_mask):
        action_scores = self.forward(states, actions, returns2go, timesteps, pad_mask) # action_scores.shape: [batch_size=1, K, a_dim]
        pred_action_scores = action_scores[:, action_idx] # pred_action_scores.shape: [batch_size=1, a_dim]
        pred_action = pred_action_scores.squeeze(0) # pred_action.shape: [a_dim]
        return pred_action


# utility function to init decision transformer
def init_decision_transformer(d_model, d_k, d_v, d_ff, n_heads, n_layers, o_dim, a_dim, max_action, num_actions, max_return, max_timestep, K, dropout, device):
    seq_len = 3 * K
    embeddings = Embeddings(d_model, K, o_dim, a_dim, num_actions, max_return, max_timestep, dropout)
    attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
    ff = FeedForward(d_model, d_ff, dropout)
    encoder_layer = EncoderLayer(deepcopy(attn), deepcopy(ff), d_model, dropout)
    encoder = Encoder(encoder_layer, n_layers, d_model)
    model = DecisionTransformer(embeddings, encoder, d_model, num_actions, a_dim, max_action, seq_len, device)
    # initialize params - Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# utility function to calculate loss (CrossEntropyLoss as action is discrete in this env)
def calculate_cross_entropy_loss(minibatch_states, minibatch_actions, minibatch_returns2go, minibatch_timesteps, minibatch_pad_mask, dt_model, a_dim):
    action_scores = dt_model(minibatch_states, minibatch_actions, minibatch_returns2go, minibatch_timesteps, minibatch_pad_mask) # action_scores.shape: [batch_size, K, num_actions]
    action_scores = action_scores.permute(0, 2, 1) # action_scores.shape: [batch_size, num_actions, K]
    action_targets = minibatch_actions.long() # action_targets.shape: [batch_size, K, a_dim]
    if a_dim == 1:
        action_targets = action_targets.squeeze(-1) # action_targets.shape: [batch_size, K]
    criterion = nn.CrossEntropyLoss(reduction='mean')
    loss = criterion(action_scores, action_targets)
    return loss

# utility function to calculate loss (MSE loss for continuous action)
def calculate_mse_loss(minibatch_states, minibatch_actions, minibatch_returns2go, minibatch_timesteps, minibatch_pad_mask, dt_model, a_dim):
    action_scores = dt_model(minibatch_states, minibatch_actions, minibatch_returns2go, minibatch_timesteps, minibatch_pad_mask) # action_scores.shape: [batch_size, K, a_dim]
    action_targets = minibatch_actions # action_targets.shape: [batch_size, K, a_dim]
    criterion = nn.MSELoss(reduction='mean')
    loss = criterion(action_scores, action_targets)
    return loss


# utility function to sample minibatch of trajectories (each of length K)
def sample_minibatch(K, batch_size, dataset_states, dataset_actions, dataset_returns2go, dataset_timesteps, a_dim, env, pad_state, pad_return2go, device):
    minibatch_states, minibatch_actions, minibatch_returns2go, minibatch_timesteps = [], [], [], []
    N = len(dataset_states)
    # init pad mask
    minibatch_pad_mask = torch.zeros(batch_size, 3*K)

    batch_index = -1
    while len(minibatch_states) < batch_size:
        batch_index += 1
        i = np.random.randint(N)
        states, actions, returns2go, timesteps = dataset_states[i], dataset_actions[i], dataset_returns2go[i], dataset_timesteps[i]
        if len(states) > K:
            j = np.random.randint(len(states) - K + 1)
            states = states[j : j+K]
            actions = actions[j : j+K]
            returns2go = returns2go[j : j+K]
            timesteps = timesteps[j : j+K]
        else:
            # add to pad mask
            for j in range(len(states), K):
                minibatch_pad_mask[batch_index][3*j] = 1
                minibatch_pad_mask[batch_index][3*j + 1] = 1
                minibatch_pad_mask[batch_index][3*j + 2] = 1
            # add padding
            while len(states) < K:
                states.append(pad_state) # sT = sT (absorbing state)
                actions.append(env.action_space.sample()) # aT = pad_action (to avoid incurring loss over this action)
                returns2go.append(pad_return2go) # RT = 0
                timesteps.append(timesteps[-1] + 1)
        # append to minibatch
        minibatch_states.append(states)
        minibatch_actions.append(actions)
        minibatch_returns2go.append(returns2go)
        minibatch_timesteps.append(timesteps)

    minibatch_states = torch.from_numpy(np.array(minibatch_states)).float().to(device) # minibatch_states.shape: [batch_size, K, o_dim]
    minibatch_actions = torch.from_numpy(np.array(minibatch_actions)).float().to(device) # minibatch_actions.shape: [batch_size, K]
    minibatch_returns2go = torch.from_numpy(np.array(minibatch_returns2go)).float().to(device) # minibatch_returns2go.shape: [batch_size, K]
    minibatch_returns2go = minibatch_returns2go.unsqueeze(-1) # minibatch_returns2go.shape: [batch_size, K, 1]
    minibatch_timesteps = torch.from_numpy(np.array(minibatch_timesteps)).long().to(device) # minibatch_timesteps.shape: [batch_size, K]
    # minibatch_timesteps = minibatch_timesteps.unsqueeze(-1) # minibatch_timesteps.shape: [batch_size, K, 1]
    # if a_dim == 1:
    #     minibatch_actions = minibatch_actions.unsqueeze(-1) # minibatch_actions.shape: [batch_size, K, 1]
    minibatch_pad_mask = minibatch_pad_mask.to(device)
    # print('\n------------------')
    # print('minibatch_states.shape: ', minibatch_states.shape)
    # print('minibatch_actions.shape: ', minibatch_actions.shape)
    # print('minibatch_returns2go.shape: ', minibatch_returns2go.shape)
    # print('minibatch_timesteps.shape: ', minibatch_timesteps.shape)
    return minibatch_states, minibatch_actions, minibatch_returns2go, minibatch_timesteps, minibatch_pad_mask


# utility function for eps-greedy action
def eps_greedy_action(action, epsilon, a_dim, max_action):
    if np.random.rand(1) < epsilon:
        action = np.random.rand(a_dim,) * max_action
    return action


### main
if __name__ == '__main__':
    # hyperparams
    d_model = 512
    d_k = 64
    d_v = 64
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    dropout = 0.1
    batch_size = 4
    lr = 1e-4
    num_epochs = 20000
    eval_freq = 40
    max_dataset_size = 20
    eval_iters = 10
    init_random_episodes = 10
    init_expert_episodes = 0
    random_seed = 1010
    render_final_episodes = True

    K = 16 # K in the paper, such that transformer input seq_len = 3K (set huetistically for now)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load environment
    env = gym.make('Pendulum-v1')
    a_dim = env.action_space.shape[0]
    o_dim = env.observation_space.shape[0]
    max_action = float(env.action_space.high[0])
    num_actions = 1

    max_timestep = env._max_episode_steps
    max_return = 0

    expert_threshold_return = -5
    dataset_max_return = -float('inf')

    eval_target_return = max_return
    mean_eval_return_cap = -float('inf')
    # best_action_trace = []

    # fix the pad action to be an out of domain action - using different actions or an in-domain action as pad_action will confuse the transformer (similar to not having a fixed out_of_vocab pad token and using arbitrary vocab tokens as pad_token during translation)
    # pad_action = num_actions
    # num_actions += 1
    # similarly fix the pad state to be the terminal state - using different states as pad_states will confuse the transformer (similar to not having a fixed pad token and using arbitrary vocab token as pad_token during translation)
    pad_state = env.reset() # we'll change this to the terminal state
    # fix pad_return2go = 0
    pad_return2go = 0

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    env.action_space.seed(random_seed)

    # epsilon schedule
    epsilon_schedule = np.ones(num_epochs) * 0
    epsilon_schedule[:int(num_epochs * 0.8)] = np.linspace(1., 0., int(num_epochs * 0.8))

    # dataset container
    # note that trajectory is denoted as <R0, s0, a0>; <R1, s1, a1>; ...; <RT, sT, aT> - where aT is a pad_action
    dataset_states, dataset_actions, dataset_returns2go, dataset_timesteps = [], [], [], []

    # seed dataset with some random trajectories
    # for ep in range(init_random_episodes):
    print('Collecting data trajectories...')
    pbar = tqdm(total=init_random_episodes + init_expert_episodes)
    data_items = 0
    while data_items < init_random_episodes + init_expert_episodes:
        ep_states, ep_actions, ep_rewards, ep_timesteps = [], [], [0.], [] # first reward should be 0
        state = env.reset()
        timestep = 0 # timesteps are zero indexed in this implementation

        while True:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            ep_states.append(state) # s_t
            ep_actions.append(action) # a_t
            ep_rewards.append(reward) # r_t+1
            ep_timesteps.append(timestep) # t
            timestep += 1

            # termination handling
            if not done:
                state = next_state
            else:
                # append terminal transition
                pad_state = next_state # setting pad_state to terminal state (TODO: what if there are multiple terminal states?)
                ep_states.append(pad_state) # s_T
                ep_actions.append(env.action_space.sample()) # a_T
                ep_timesteps.append(timestep) # T
                break

        # episode / trajectory terminated - calculate returns2go
        ep_return = sum(ep_rewards)
        ep_returns2go = []
        curr_return2go = ep_return
        for i in range(len(ep_rewards)):
            curr_return2go -= ep_rewards[i]
            ep_returns2go.append(curr_return2go)

        # append to dataset
        if len(dataset_states) < (init_random_episodes): # random episode
            dataset_states.append(ep_states)
            dataset_actions.append(ep_actions)
            dataset_returns2go.append(ep_returns2go)
            dataset_timesteps.append(ep_timesteps)
            if ep_returns2go[0] > dataset_max_return:
                dataset_max_return = ep_returns2go[0]
            pbar.update(1)
            data_items += 1
        else: # expert episode
            if ep_return > expert_threshold_return:
                expert_dataset_states.append(ep_states)
                expert_dataset_actions.append(ep_actions)
                expert_dataset_returns2go.append(ep_returns2go)
                expert_dataset_timesteps.append(ep_timesteps)
                if ep_returns2go[0] > dataset_max_return:
                    dataset_max_return = ep_returns2go[0]
                pbar.update(1)
                data_items += 1


    pbar.close()
    print('------------ dataset_max_return: ', dataset_max_return)
    print('Data collected. Starting training...')

    # containers for storing statistics for plotting
    result_loss = []
    result_return = []
    # print('------------- max_return: ', max_return)

    # init model (decision_transformer)
    dt_model = init_decision_transformer(d_model, d_k, d_v, d_ff, n_heads, n_layers, o_dim, a_dim, max_action, num_actions, max_return, max_timestep, K, dropout, device).to(device)

    # init optimizer
    optimizer = torch.optim.AdamW(params=dt_model.parameters(), lr=lr)

    ## train loop
    for ep in tqdm(range(num_epochs)):

        # resize dataset - select most recent entries
        dataset_states = dataset_states[-max_dataset_size:]
        dataset_actions = dataset_actions[-max_dataset_size:]
        dataset_returns2go = dataset_returns2go[-max_dataset_size:]
        dataset_timesteps = dataset_timesteps[-max_dataset_size:]
        #
        # # pre-pend expert dataset
        # dataset_states = expert_dataset_states + dataset_states
        # dataset_actions = expert_dataset_actions + dataset_actions
        # dataset_returns2go = expert_dataset_returns2go + dataset_returns2go
        # dataset_timesteps = expert_dataset_timesteps + dataset_timesteps

        # sample minibatch of trajectories (each of length K)
        minibatch_states, minibatch_actions, minibatch_returns2go, minibatch_timesteps, minibatch_pad_mask = sample_minibatch(K, batch_size, dataset_states, dataset_actions, dataset_returns2go, dataset_timesteps, a_dim, env, pad_state, pad_return2go, device)
        # print('\n------------------------')
        # print('\nminibatch_actions:\n', minibatch_actions)
        # print('\nminibatch_returns2go:\n', minibatch_returns2go)
        # print('\minibatch_timesteps:\n', minibatch_timesteps)
        # print('\minibatch_pad_mask:\n', minibatch_pad_mask)


        # calculate loss (mse_loss as action is continuous in this env) and take gradient step
        loss = calculate_mse_loss(minibatch_states, minibatch_actions, minibatch_returns2go, minibatch_timesteps, minibatch_pad_mask, dt_model, a_dim)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep+1) % eval_freq == 0:
            # print loss
            # print('loss: {:.3f}'.format(loss.item()))
            result_loss.append(loss.item())

            # eval
            dt_model.eval()

            with torch.no_grad():

                eval_returns = []
                for iter in range(eval_iters):

                    state = env.reset()
                    timestep = 0
                    ep_states, ep_actions, ep_rewards, ep_timesteps = [], [], [0.], [] # first reward should be 0
                    eval_states, eval_actions, eval_returns2go, eval_timesteps = [state], [env.action_space.sample()], [eval_target_return], [0] # target_return = max_return; pad_action is a placeholder for predicted action
                    action_idx = 0 # index of the predicted action in the transformer output

                    while True:

                        if render_final_episodes and (ep > num_epochs - 10 * eval_freq):
                            env.render()

                        # pad trajectory if needed (and its necessary to add a pad_mask here so that the information from pad_tokens doesn't propogate in the transformer)
                        pad_mask = None
                        if len(eval_states) < K:
                            # create pad mask - required during eval
                            pad_mask = torch.zeros(1, 3*K).to(device)
                            for j in range(len(eval_states), K):
                                pad_mask[0][3*j] = 1
                                pad_mask[0][3*j + 1] = 1
                                pad_mask[0][3*j + 2] = 1
                            # add pad tokens
                            while len(eval_states) < K:
                                eval_states.append(pad_state)
                                eval_actions.append(env.action_space.sample())
                                eval_returns2go.append(pad_return2go)
                                eval_timesteps.append(eval_timesteps[-1] + 1)

                        # convert to tensors to input the transformer
                        input_states = torch.from_numpy(np.array([eval_states])).float().to(device)
                        input_actions = torch.from_numpy(np.array([eval_actions])).float().to(device)
                        input_returns2go = torch.from_numpy(np.array([eval_returns2go])).float().to(device)
                        input_returns2go = input_returns2go.unsqueeze(-1)
                        input_timesteps = torch.from_numpy(np.array([eval_timesteps])).long().to(device)
                        # input_timesteps = input_timesteps.unsqueeze(-1)
                        # if a_dim == 1:
                        #     input_actions = input_actions.unsqueeze(-1)

                        # get predicted action
                        action = dt_model.predict(input_states, input_actions, input_returns2go, input_timesteps, action_idx, pad_mask)
                        action = action.cpu().numpy()

                        # avoid pad_action - take random action if predicted_action is pad_action (this should not happen as loss decreases and we converge to the correct policy / behaviour)
                        # if action_scalar == pad_action:
                        #     action_scalar = env.action_space.sample()

                        # action exploration - eps_greedy
                        epsilon = epsilon_schedule[ep]
                        action = eps_greedy_action(action, epsilon, a_dim, max_action)
                        # take action
                        next_state, reward, done, _ = env.step(action)

                        ep_states.append(state) # s_t
                        ep_actions.append(action) # a_t
                        ep_rewards.append(reward) # r_t+1
                        ep_timesteps.append(timestep) # t

                        # for next step
                        if action_idx < (K-1):
                            action_idx += 1 # for next step
                            eval_states = eval_states[:action_idx] + [next_state]
                            eval_actions = eval_actions[:action_idx-1] + [action] + [env.action_space.sample()]
                            eval_returns2go = eval_returns2go[:action_idx]
                            eval_returns2go = eval_returns2go + [eval_returns2go[-1] - reward]
                            eval_timesteps = eval_timesteps[:action_idx]
                            eval_timesteps = eval_timesteps + [eval_timesteps[-1] + 1]
                        else:
                            eval_states = eval_states[1:] + [next_state]
                            eval_actions = eval_actions[1:-1] + [action] + [env.action_space.sample()]
                            eval_returns2go = eval_returns2go[1:]
                            eval_returns2go = eval_returns2go + [eval_returns2go[-1] - reward]
                            eval_timesteps = eval_timesteps[1:]
                            eval_timesteps = eval_timesteps + [eval_timesteps[-1] + 1]

                        timestep += 1
                        # termination handling
                        if not done:
                            state = next_state
                        else:
                            # append terminal transition
                            ep_states.append(next_state) # s_T
                            ep_actions.append(env.action_space.sample()) # a_T
                            ep_timesteps.append(timestep) # T
                            break

                    # episode / trajectory terminated - calculate returns2go
                    ep_return = sum(ep_rewards)
                    eval_returns.append(ep_return)
                    ep_returns2go = []
                    curr_return2go = ep_return
                    for i in range(len(ep_rewards)):
                        curr_return2go -= ep_rewards[i]
                        ep_returns2go.append(curr_return2go)

                    # append eval trajectories to dataset - if the return of the trajectory is greater than mean_eval_return_cap
                    # if ep_return > mean_eval_return_cap:
                    dataset_states.append(ep_states)
                    dataset_actions.append(ep_actions)
                    dataset_returns2go.append(ep_returns2go)
                    dataset_timesteps.append(ep_timesteps)
                    print('len(dataset_states):{} \t mean_eval_return_cap:{:.3f} \t ep_return:{:.3f}'.format(len(dataset_states), mean_eval_return_cap, ep_return))


                # calculate mean_eval_return
                mean_eval_return = sum(eval_returns) / len(eval_returns)
                # update max_return
                max_eval_return = max(eval_returns)

                if mean_eval_return_cap < mean_eval_return:
                    mean_eval_return_cap = mean_eval_return
                    print('------------- mean_eval_return_cap: ', mean_eval_return_cap)
                # append to stats for plotting
                result_return.append(mean_eval_return)


            dt_model.train()


    # plot statistics
    fig, ax = plt.subplots(1,2)

    ax[0].plot(result_loss, color='red', label='ep_loss')
    ax[0].legend()
    ax[0].set(xlabel='episode')
    # ax[0].set_ylim([0, 10])

    ax[1].plot(result_return, color='green', label='ep_return')
    ax[1].legend()
    ax[1].set(xlabel='episode')
    # ax[1].set_ylim([-50, 250])

    plt.show()
