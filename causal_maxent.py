import torch as t
from tqdm.notebook import trange
from rewards import Linear, Perceptron, MLP, GCN
import plotly.express as px
import pandas as pd
import numpy as np


def stochastic_value_iteration(transition, reward, discount, eps=1e-5):
    n_states = transition.shape[0]
    v = np.zeros(n_states)
    
    delta = np.inf
    while delta > eps:      # iterate until convergence
        v_old = v

        # compute state-action values 
        q = discount * (transition @ v)

        # compute state values
        v = reward + np.average(q, axis=0)

        # compute maximum delta
        delta = np.max(np.abs(v_old - v))

    return v

def expected_svf_from_policy(p_transition, p_initial, terminal, p_action, eps=1e-5):
    """
    Params
    -----
    p_transition: (N x N) Transition Probabilities
    p_initial: (N, 1) Probability of becoming an initial state
    terminal: (k) Indices of k initial states
    p_action: (N x N) local action probabilities
    eps: convergence threshold

    Return
    -----
    res: (N, 1) state visitation frequency
    """
    n_states, n_actions = p_transition.shape

    p_transition = p_transition.clone()
    p_transition[terminal, :] = 0.0
    delta = 1e20
    res = p_initial
    d_ = None
    save_p_action = False

    while delta > eps: 
    # for idx in range(30):
        """
        version that works, 30 March 2025
        """
        d_ = p_transition.T @ (p_action * res)
        # in_D_inv = (1/ (d_.T.sum(1) + 1))
        out_D_inv = (1/ (d_.sum(1) + 1))
        d_ = out_D_inv.view(-1, 1) * d_.float() #* in_D_inv
        d = p_initial + d_.sum(axis=1)
        delta, res = t.max(t.abs(d - res)), d

    laplacian_max_eigval = max(t.linalg.eigvals(d_).real)

    return res, laplacian_max_eigval


def local_causal_action_probabilities(
    p_transition, terminal, reward, discount, eps=1e-5
):
    """
    Params
    -----
    p_transition: (N x N) Transition Probabilities
    terminal: (k) Indices of k initial states
    reward: (N,1) Reward for each state
    discount: scalar
    eps: convergence threshold

    Return
    -----
    res: (N, N) policy
    """
    n_states, n_actions = p_transition.shape

    if len(terminal) == n_states:
        reward_terminal = t.tensor(terminal).float()
    else:
        reward_terminal = -1e20 * t.ones(n_states)
        reward_terminal[terminal] = 0.0
        
    v = -1e20 * t.ones(n_states)
    delta = 1e20
    
    # for _ in range(30):
    while delta > eps:
        v_old = v
        v = reward_terminal
        q = reward + discount * p_transition * v_old
        v = t.cat([v.unsqueeze(1),q],1)
        v = t.logsumexp(v, 1)
        delta = t.max(t.abs(v - v_old))
        
    return t.exp((q.T - v).T)

def irl_causal(
    features,
    model,
    optim,
    p_transition,
    p_initial,
    terminal,
    scheduler=None,
    discount=0.7,
    n_epochs=30,
    device="mps",
    eps=1e-4,
    eps_svf=1e-5,
    eps_lap=1e-5,
):
    model.train()
    delta = []
    laplacian_max_eigval = []
    best_loss = 1e20
    best_reward = None

    expert_svf = features.sum(1)
    for i in trange(n_epochs):
    # for i in range(n_epochs):
        rewards = model(features).flatten()
        p_action = local_causal_action_probabilities(
            p_transition, terminal, rewards.detach(), discount
        )
        expected_svf, max_eigval = expected_svf_from_policy(
            p_transition, p_initial, terminal, p_action
        )
        laplacian_max_eigval.append(max_eigval)
        loss_grad = expert_svf - expected_svf.float()
        curr_loss = loss_grad.detach().max().cpu().abs()
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_reward = rewards
        delta.append(curr_loss)
        rewards.backward(-loss_grad)

        if ((t.tensor(delta)[-50:]) <= curr_loss).all() and (i > 60):
            print("early stop!")
            break
        optim.step()
        if scheduler is not None:
            scheduler.step()
        optim.zero_grad()
    fig = px.line(x=list(range(len(delta))), y=delta, title="Loss")
    fig.show()
    fig = px.line(x=list(range(len(laplacian_max_eigval))), y=laplacian_max_eigval, title="Principal Eigenvalue of Intermediate Policy")
    fig.show()
    return best_reward.detach().numpy(), delta, laplacian_max_eigval
