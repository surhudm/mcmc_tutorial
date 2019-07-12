#!/usr/bin/env python3

import numpy as np
import emcee
import time

#
# Class for a Multidimensional-Gaussian distribution
#
class gaussian_md:

    def __init__(self, means, cov):
        self.mu = means
        self.icov = np.linalg.inv(cov)

    #
    # This function should return log(Prob) = -Chisq/2 upto a additive constant
    # For a Gaussian distribution this will be -1/2.(x-mu)^T Cinv (x-mu)
    #
    def lnprob(self, x):
        Delta = (x-self.mu)
        # Test a slow calculation by introducing a sleep command
        # time.sleep(0.1)
        return -np.dot(Delta, np.dot(self.icov, Delta))/2.0

    # Get a emcee sampler with the object
    def getsampler(self, nwalkers, ndim):
        return emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, args=[])

if __name__ == "__main__":

    # Initialize random number seed
    np.random.seed(10)

    # Define number of dimensions
    ndim = 50
    
    # Define random means and some covariance (first just a diagonal covariance)
    means = np.random.rand(ndim) * 5.0
    cov = np.diag(np.linspace(5.0, 10.0, ndim))

    aa = gaussian_md(means, cov)

    # Initialize tons of walkers
    nwalkers = 320
    p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))

    # Initialize the sampler
    sampler = aa.getsampler(nwalkers, ndim)

    # Perform an initial burn-in phase, clear out the sampler, but store the
    # final locations in pos, the value of lnprob and random number state
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()

    Nsamples = 10000
    sampler.run_mcmc(pos, Nsamples)

    # Now the sampler will have chain values store in sampler.flatchain, let us first see their shape
    print("Shape of sampler.chain", np.shape(sampler.chain))
    print("Shape of sampler.flatchain", np.shape(sampler.flatchain))

    '''
    import matplotlib.pyplot as pl
    import corner

    for i in range(ndim):
        ax = pl.subplot(3, 3, i+1)
        ax.hist(sampler.flatchain[:,i], 100, color="k", histtype="step")
        ax.axvline(aa.mu[i])

    pl.savefig("Distributions.png")
    pl.clf()

    for i in range(ndim):
        ax = pl.subplot(3, 3, i+1)
        ax.plot(np.arange(Nsamples), sampler.chain[0, :,i])
        ax.axhline(aa.mu[i], color="k")

    pl.savefig("Chains.png")

    fig = corner.corner(sampler.chain.reshape(-1, ndim))
    fig.savefig("Triangle.png")
    '''
