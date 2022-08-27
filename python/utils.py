import torch
import pandas as pd


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
    return -torch.mean(torch.distributions.beta.Beta(alpha, beta).log_prob(v))


def load_csv(file_name):
    data = pd.read_csv(file_name, delimiter=',')
    data = data.to_numpy()
    belief_state = data[:, : -3]
    action_value = data[:, -3:-2]
    action_count = data[:, -2:-1]
    step = data[:, -1:]

    return belief_state, action_value, action_count, step
