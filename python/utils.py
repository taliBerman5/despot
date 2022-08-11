import torch


def bsa_loss(output, target):
    """output is what we predict (v and n)
    target is what we get, now consisting of values and sample sizes.
    """
    logitmean, logn = output[:, 0], output[:, 1]
    mean = torch.special.expit(logitmean)
    n = torch.exp(logn)
    v, ss = target

    # We account for ss in the effective sample size
    neff = 1/(1/n + 1/ss)
    alpha = mean*neff
    beta = neff - alpha
    return -torch.mean(torch.distributions.beta.Beta(alpha, beta).log_prob(v))
