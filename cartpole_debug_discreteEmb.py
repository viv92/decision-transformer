### Program implementing Decision Transformer on CartPole-v1 environment

## Features:
# 1. Solve RL problems using a causal transformer encoder by reformulating them as sequence prediction problem
# 2. This implementation does not use a purely offline-RL setting, as we append the new trajectories (obtained during evaluation) to the dataset

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
    def __init__(self, d_model, K, o_dim, a_dim, num_actions, max_return, dropout):
        super().__init__()
        self.state_emb = nn.Linear(o_dim, d_model, bias=False)
        self.action_emb = nn.Embedding(num_actions, d_model)
        self.returns2go_emb = nn.Embedding(int(max_return)+1, d_model)
        self.pos_emb = nn.Parameter(torch.randn(K, d_model)) # learnt positional embeddings
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, states, actions, returns2go): # shape: [batch_size, K, dim]
        batch_size = states.shape[0]
        # get embeddings
        states_emb = self.state_emb(states)
        actions_emb = self.action_emb(actions)
        returns2go_emb = self.returns2go_emb(returns2go)
        # add positional embeddings
        pos_emb = self.pos_emb
        pos_emb = pos_emb.unsqueeze(0) # pos_emb.shape: [1, K, d_model]
        pos_emb = pos_emb.expand(batch_size, -1, -1) # pos_emb.shape: [batch_size, K, d_model]
        states_emb = self.dropout( self.norm(states_emb + pos_emb) )
        actions_emb = self.dropout( self.norm(actions_emb + pos_emb) )
        returns2go_emb = self.dropout( self.norm(returns2go_emb + pos_emb) )
        # concat states, actions, returns2go into a single trajectory of length 3*K
        traj = torch.stack((states_emb, actions_emb, returns2go_emb), dim=1) # traj.shape: [batch_size, K, 3, d_model]
        traj = torch.flatten(traj, start_dim=-3, end_dim=-2) # traj.shape: [batch_size, 3K, d_model]
        return traj


# class implementing the Decision Transformer
class DecisionTransformer(nn.Module):
    def __init__(self, embeddings, encoder, d_model, num_actions, seq_len, device):
        super().__init__()
        self.embeddings = embeddings
        self.encoder = encoder
        self.action_proj = nn.Linear(d_model, num_actions, bias=False)
        self.action_idx = torch.arange(start=2, end=seq_len, step=3).to(device)
        self.device = device
    # function for forward prop through the transformer encoder and get action scores
    def forward(self, states, actions, returns2go): # shape: [batch_size, K, dim]
        # convert raw data to embeddings
        traj = self.embeddings(states, actions, returns2go) # traj.shape: [batch_size, 3K, d_model]
        # get causal mask
        batch_size, seq_len = traj.shape[0], traj.shape[1]
        causal_mask = subsequent_mask((batch_size, seq_len)).to(self.device)
        # forward prop through transformer encoder
        out = self.encoder(traj, mask_causal=causal_mask) # out.shape: [batch_size, 3K, d_model]
        # extract predicted actions scores
        action_scores = out[:, self.action_idx] # action_scores.shape: [batch_size, K, d_model]
        # project action_scores to num_actions
        action_scores = self.action_proj(action_scores) # action_scores.shape: [batch_size, K, num_actions]
        return action_scores
    # function to autoregressively predict an action
    def predict(self, states, actions, returns2go, action_idx):
        action_scores = self.forward(states, actions, returns2go) # action_scores.shape: [batch_size=1, K, num_actions]
        pred_action_scores = action_scores[:, action_idx] # pred_action_scores.shape: [batch_size=1, num_actions]
        pred_action_probs = F.softmax(pred_action_scores, dim=-1).squeeze(0) # pred_action_probs.shape: [num_actions]
        pred_action = torch.argmax(pred_action_probs)
        return pred_action


# utility function to init decision transformer
def init_decision_transformer(d_model, d_k, d_v, d_ff, n_heads, n_layers, o_dim, a_dim, num_actions, K, dropout, max_return, device):
    seq_len = 3 * K
    embeddings = Embeddings(d_model, K, o_dim, a_dim, num_actions, max_return, dropout)
    attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
    ff = FeedForward(d_model, d_ff, dropout)
    encoder_layer = EncoderLayer(deepcopy(attn), deepcopy(ff), d_model, dropout)
    encoder = Encoder(encoder_layer, n_layers, d_model)
    model = DecisionTransformer(embeddings, encoder, d_model, num_actions, seq_len, device)
    # initialize params - Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# utility function to calculate loss (CrossEntropyLoss as action is discrete in this env)
def calculate_cross_entropy_loss(minibatch_states, minibatch_actions, minibatch_returns2go, dt_model, a_dim):
    action_scores = dt_model(minibatch_states, minibatch_actions, minibatch_returns2go) # action_scores.shape: [batch_size, K, num_actions]
    action_scores = action_scores.permute(0, 2, 1) # action_scores.shape: [batch_size, num_actions, K]
    action_targets = minibatch_actions # action_targets.shape: [batch_size, K, a_dim]
    if a_dim == 1:
        action_targets = action_targets.squeeze(-1) # action_targets.shape: [batch_size, K]
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loss = criterion(action_scores, action_targets)
    return loss


# utility function to sample minibatch of trajectories (each of length K)
def sample_minibatch(K, batch_size, dataset_states, dataset_actions, dataset_returns2go, a_dim, pad_action, device):
    minibatch_states, minibatch_actions, minibatch_returns2go = [], [], []
    N = len(dataset_states)
    while len(minibatch_states) < batch_size:
        i = np.random.randint(N)
        states, actions, returns2go = dataset_states[i], dataset_actions[i], dataset_returns2go[i]
        if len(states) > K:
            j = np.random.randint(len(states) - K + 1)
            states = states[j : j+K]
            actions = actions[j : j+K]
            returns2go = returns2go[j : j+K]
        else:
            while len(states) < K: # pad
                states.append(states[-1]) # sT = sT (absorbing state)
                actions.append(pad_action) # aT = pad_action (to avoid incurring loss over this action)
                returns2go.append(returns2go[-1]) # RT = 0
        # append to minibatch
        minibatch_states.append(states)
        minibatch_actions.append(actions)
        minibatch_returns2go.append(returns2go)

    minibatch_states = torch.from_numpy(np.array(minibatch_states)).to(device) # minibatch_states.shape: [batch_size, K, o_dim]
    minibatch_actions = torch.from_numpy(np.array(minibatch_actions)).long().to(device) # minibatch_actions.shape: [batch_size, K]
    minibatch_returns2go = torch.from_numpy(np.array(minibatch_returns2go)).long().to(device) # minibatch_returns2go.shape: [batch_size, K]
    # minibatch_returns2go = minibatch_returns2go.unsqueeze(-1) # minibatch_returns2go.shape: [batch_size, K, 1]
    # if a_dim == 1:
    #     minibatch_actions = minibatch_actions.unsqueeze(-1) # minibatch_actions.shape: [batch_size, K, 1]
    return minibatch_states, minibatch_actions, minibatch_returns2go


# utility function for eps-greedy action
def eps_greedy_action(action, epsilon, num_actions):
    if np.random.rand(1) < epsilon:
        action = np.random.randint(num_actions)
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
    batch_size = 256
    lr = 1e-4
    num_epochs = 2000
    eval_freq = 20
    # eval_iters = 1
    init_random_episodes = 10000
    random_seed = 1010

    K = 16 # K in the paper, such that transformer input seq_len = 3K (set huetistically for now)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load environment
    env = gym.make('CartPole-v1')
    num_actions = env.action_space.n # used as num_classes for CrossEntropyLoss
    a_dim = 1 # actions are discrete int
    o_dim = env.observation_space.shape[0]

    max_return = 0 #500
    min_return = 0 #50
    best_action_trace = []

    # fix one of the actions as the pad_action to avoid incurring loss on the action taken at terminal state
    pad_action = env.action_space.sample()
    # # ensure K doesn't exceed _max_episode_steps - to avoid unnecessary padding everytime
    # assert K < env._max_episode_steps

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
    dataset_states, dataset_actions, dataset_returns2go = [], [], []

    # seed dataset with some random trajectories
    # for ep in range(init_random_episodes):
    while len(dataset_states) < init_random_episodes:
        ep_states, ep_actions, ep_rewards = [], [], [0.] # first reward should be 0
        state = env.reset()

        while True:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            ep_states.append(state) # s_t
            ep_actions.append(action) # a_t
            ep_rewards.append(reward) # r_t+1

            # termination handling
            if not done:
                state = next_state
            else:
                # append terminal transition
                ep_states.append(next_state) # s_T
                ep_actions.append(pad_action) # a_T
                break

        # episode / trajectory terminated - calculate returns2go
        ep_return = sum(ep_rewards)
        ep_returns2go = []
        curr_return2go = ep_return
        for i in range(len(ep_rewards)):
            curr_return2go -= ep_rewards[i]
            ep_returns2go.append(curr_return2go)

        if max_return < ep_return:
            max_return = ep_return
            best_action_trace = ep_actions

        # append to dataset
        if ep_return > min_return:
            dataset_states.append(ep_states)
            dataset_actions.append(ep_actions)
            dataset_returns2go.append(ep_returns2go)

    # containers for storing statistics for plotting
    result_loss = []
    result_return = []
    print('------------- max_return: ', max_return)

    # init model (decision_transformer)
    dt_model = init_decision_transformer(d_model, d_k, d_v, d_ff, n_heads, n_layers, o_dim, a_dim, num_actions, K, dropout, max_return, device).to(device)

    # init optimizer
    optimizer = torch.optim.AdamW(params=dt_model.parameters(), lr=lr)

    ## train loop
    for ep in tqdm(range(num_epochs)):

        # sample minibatch of trajectories (each of length K)
        minibatch_states, minibatch_actions, minibatch_returns2go = sample_minibatch(K, batch_size, dataset_states, dataset_actions, dataset_returns2go, a_dim, pad_action, device)

        # calculate loss (CrossEntropyLoss as action is discrete in this env) and take gradient step
        loss = calculate_cross_entropy_loss(minibatch_states, minibatch_actions, minibatch_returns2go, dt_model, a_dim)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep+1) % (num_epochs // eval_freq) == 0:
            # print loss
            print('loss: {:.3f}'.format(loss.item()))
            result_loss.append(loss.item())

            # eval
            dt_model.eval()

            with torch.no_grad():

                state = env.reset()
                ep_states, ep_actions, ep_rewards = [], [], [0.] # first reward should be 0
                eval_states, eval_actions, eval_returns2go = [state], [pad_action], [max_return] # target_return = max_return; pad_action is a placeholder for predicted action
                action_idx = 0 # index of the predicted action in the transformer output

                while True:

                    # pad trajectory if needed (doesn't matter what we pad with because it doesn't affect the predicted_action at action_idx due to the causal mask)
                    while len(eval_states) < K:
                        eval_states.append(state)
                        eval_actions.append(pad_action)
                        eval_returns2go.append(max_return)

                    # convert to tensors to input the transformer
                    input_states = torch.from_numpy(np.array([eval_states])).to(device)
                    input_actions = torch.from_numpy(np.array([eval_actions])).long().to(device)
                    input_returns2go = torch.from_numpy(np.array([eval_returns2go])).long().to(device)
                    # input_returns2go = input_returns2go.unsqueeze(-1)
                    # if a_dim == 1:
                    #     input_actions = input_actions.unsqueeze(-1)

                    # get predicted action
                    action = dt_model.predict(input_states, input_actions, input_returns2go, action_idx)
                    action_scalar = action.item()
                    # action exploration - eps_greedy
                    epsilon = epsilon_schedule[ep]
                    # action_scalar = eps_greedy_action(action_scalar, epsilon, num_actions)
                    # take action
                    next_state, reward, done, _ = env.step(action_scalar)

                    ep_states.append(state) # s_t
                    ep_actions.append(action_scalar) # a_t
                    ep_rewards.append(reward) # r_t+1

                    # for next step
                    if action_idx < (K-1):
                        action_idx += 1 # for next step
                        eval_states = eval_states[:action_idx] + [next_state]
                        eval_actions = eval_actions[:action_idx-1] + [action_scalar] + [pad_action]
                        eval_returns2go = eval_returns2go[:action_idx]
                        eval_returns2go = eval_returns2go + [eval_returns2go[-1] - reward]
                    else:
                        eval_states = eval_states[1:] + [next_state]
                        eval_actions = eval_actions[1:-1] + [action_scalar] + [pad_action]
                        eval_returns2go = eval_returns2go[1:]
                        eval_returns2go = eval_returns2go + [eval_returns2go[-1] - reward]

                    # termination handling
                    if not done:
                        state = next_state
                    else:
                        # append terminal transition
                        ep_states.append(next_state) # s_T
                        ep_actions.append(pad_action) # a_T
                        break

                # episode / trajectory terminated - calculate returns2go
                ep_return = sum(ep_rewards)
                ep_returns2go = [ep_return - x for x in ep_rewards]

                print('ep_return: ', ep_return)
                result_return.append(ep_return)

                print('action_trace_comparison:')
                action_trace_comparison = [(best_action_trace[i], ep_actions[i]) for i in range(len(ep_actions))]
                print(action_trace_comparison)


            dt_model.train()


    # plot statistics
    fig, ax = plt.subplots(1,2)

    ax[0].plot(result_loss, color='red', label='ep_loss')
    ax[0].legend()
    ax[0].set(xlabel='episode')

    ax[1].plot(result_return, color='green', label='ep_return')
    ax[1].legend()
    ax[1].set(xlabel='episode')

    plt.show()
