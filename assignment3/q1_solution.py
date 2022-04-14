from inspect import trace
import math
import this
import numpy as np
import torch


def log_likelihood_bernoulli(mu, target):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Bernoulli random variables p(x=1).
    :param target: (FloatTensor) - shape: (batch_size x input_size) - Target samples (binary values).
    :return: (FloatTensor) - shape: (batch_size,) - log-likelihood of target samples on the Bernoulli random variables.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    target = target.view(batch_size, -1)
    un= target*torch.log(mu)
    deux = torch.log(1-mu)*(1-target) 
    troi = torch.add(un,deux)
    ans = torch.sum(troi,1)
    return ans


def log_likelihood_normal(mu, logvar, z):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Normal distributions.
    :param logvar: (FloatTensor) - shape: (batch_size x input_size) - The log variance of Normal distributions.
    :param z: (FloatTensor) - shape: (batch_size x input_size) - Target samples.
    :return: (FloatTensor) - shape: (batch_size,) - log probability of the sames on the given Normal distributions.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    z = z.view(batch_size, -1)
    n = mu.size(1)
    var =  torch.exp(logvar)
    difCarre = ((z-mu)**2)/var
    nll = torch.sum((-torch.log(torch.sqrt(2*math.pi*var))-0.5*difCarre),1)
    return nll


def log_mean_exp(y):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param y: (FloatTensor) - shape: (batch_size x sample_size) - Values to be evaluated for log_mean_exp. For example log proababilies
    :return: (FloatTensor) - shape: (batch_size,) - Output for log_mean_exp.
    """
    # init
    batch_size = y.size(0)
    sample_size = y.size(1) 
    # log_mean_exp
    a_i = torch.unsqueeze(torch.max(y,dim=1).values,1)
    expon = torch.exp(y - a_i)
    logMeanExp = torch.log(torch.mean(expon,dim = 1))
    return torch.unsqueeze(logMeanExp,1) + a_i


def kl_gaussian_gaussian_analytic(mu_q, logvar_q, mu_p, logvar_p):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    var_q = torch.exp(logvar_q)  # most be Sigma^2
    sigma_q = torch.sqrt(var_q)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)
    var_p = torch.exp(logvar_p) # most be Sigma^2
    sigma_p = torch.sqrt(var_p)
    # kld
    '''
        Ref: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    '''
    logSigmaDiv =  torch.log(sigma_p/sigma_q)
    muDifSqr = torch.pow((mu_q-mu_p), 2)
    ans = torch.sum((logSigmaDiv + ((var_q+ muDifSqr)/(2*var_p))- 0.5),1)
    return ans
    
    

def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.
    
    *** note. ***
    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    input_size = np.prod(mu_q.size()[1:])
    mu_q = mu_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_q = logvar_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    mu_p = mu_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_p = logvar_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)

    # kld
    '''
    ref : https://stats.stackexchange.com/questions/280885/estimate-the-kullback-leibler-kl-divergence-with-monte-carlo
    '''
    var_q = torch.abs(torch.exp(logvar_q))  # most be Sigma^2
    var_p = torch.abs(torch.exp(logvar_p))  # most be Sigma^2
    x = torch.normal(mu_q,var_q) # sampling from Normal Distribution P(q)
    log_q = -torch.log(torch.sqrt(2*math.pi*var_q))-(0.5 *  ((x-mu_q)**2/var_q))
    log_p = -torch.log(torch.sqrt(2*math.pi*var_p))-( 0.5 *  ((x-mu_p)**2/var_p))
    kl_mc = torch.mean(log_q-log_p,1)
    return kl_mc
    
