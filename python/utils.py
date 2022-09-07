import torch
import pandas as pd
import numpy as np


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)


def bsa_loss(output, target):
    """output is what we predict (v and n)
    target is what we get, now consisting of values and sample sizes.
    """
    logitmean, logn = output[:, 0], output[:, 1]
    mean = torch.special.expit(logitmean)
    n = torch.exp(logn)
    v, ss = target

    # We account for ss in the effective sample size
    neff = 1 / (1 / n + 1 / ss)
    alpha = mean * neff
    beta = neff - alpha
    # alpha = torch.clip(alpha, min=0.001)
    # beta = torch.clip(beta, min=0.001)
    return -torch.mean(torch.distributions.beta.Beta(alpha, beta).log_prob(v))
    # return torch.distributions.beta.Beta(alpha, beta)


def load_csv(file_name, num_states):
    data = pd.read_csv(file_name, delimiter=',')
    data = data.to_numpy()
    belief_state = data[:, : num_states]
    action_value = data[:, num_states::2][:, :-1]
    action_count = data[:, num_states + 1::2]
    step = data[:, -1:]

    return belief_state, action_value, action_count, step


def step_values(indices, belief_state, action_value, action_count):
    belief_state_step = belief_state[indices].reshape(len(indices), belief_state.shape[1])
    action_value_step = action_value[indices].reshape(len(indices), action_value.shape[1])
    action_count_step = action_count[indices].reshape(len(indices), action_count.shape[1])
    return belief_state_step, action_value_step, action_count_step
