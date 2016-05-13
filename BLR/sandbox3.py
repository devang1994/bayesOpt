import numpy as np
from bayesNN_HMCv2 import objective


def bayes_opt(func, hWidths, precisions, vy, initial_random=2, k=0.2, num_it=20):
    '''function to do bayesOpt on and number of initial random evals
    noise is artificially added to objective function calls when training
    '''
    noise = 0.01
    ntest = 1000
    ntrain = initial_random  # number of initial random function evals
    xtrain = np.random.uniform(low=-1.0, high=1.0, size=(ntrain, 1))
    print xtrain.shape
    ytrain = func(xtrain) + np.random.randn(ntrain, 1) * noise
    print ytrain.shape
    xtest = np.linspace(-1., 2., ntest)
    xtest = xtest.reshape(ntest, 1)
    ytest = func(xtest)

    ytrain_pure = func(xtrain)
    print ytrain_pure
    print xtrain
    a = np.argmin(ytrain_pure)
    print a
    print xtrain[a]
    print ytrain_pure[a]
    # TODO use this to find the current best fit


if __name__ == '__main__':
    func = objective

    bayes_opt(func, initial_random=5, num_it=10, k=2, hWidths=[50, 50, 50], precisions=[1, 1, 1, 1], vy=100,)
