from bayesNN_HMCv2 import objective, acquisition_UCB, sampler_on_BayesNN, analyse_samples

__author__ = 'da368'
import numpy

from theano import function, shared
from theano import tensor as T
import theano
from hmc import HMC_sampler
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
import cPickle as pickle
import bayesNNexplorePostfromIp
import pylab
import objectives
import time


def readPickle():
    func_name = 'objectiveForrester'
    func_name_GP = 'GpyOPTforrester'
    totalFevals = 20

    actual_min = -1.96729
    allBvals = np.asarray([])
    times = []

    allBvalsGP = np.asarray([])
    timesGP = []

    for seed in range(1000, 1020):
        print seed
        # seed=1000
        # toDump = {'bVals': bVals, 't': time_taken, 'seed': seed, 'k': k, 'init_random': init_random}
        nameOfFile = 'pickles/seed{}BayesOptLogs{}.pkl'.format(seed, func_name)

        dumpFile = pickle.load(open(nameOfFile, "rb"))

        bVals = dumpFile['bVals']
        time_taken = dumpFile['t']

        bVals = (np.asarray(bVals)).reshape(1, -1)
        # print bVals.shape
        if (allBvals.shape[0] == 0):
            allBvals = np.vstack((bVals))
            # print 'allb shape{}'.format(allBvals.shape)
        else:
            allBvals = np.vstack((allBvals, bVals))

        times.append(time_taken)

        nameOfFileGP = 'pickles/seed{}BayesOptLogs{}.pkl'.format(seed, func_name_GP)

        dumpFile = pickle.load(open(nameOfFileGP, "rb"))

        bVals = dumpFile['bVals']
        time_taken = dumpFile['t']

        bVals = (np.asarray(bVals)).reshape(1, -1)

        print bVals.shape
        if (bVals.shape[1] != totalFevals):
            # for when the algorithm converges and does not do full iterations
            curbest = bVals[0, -1]
            numNeeded = totalFevals - bVals.shape[1]
            temp = ((np.ones(numNeeded)).reshape(1, -1)) * curbest
            # bVals

        if (allBvalsGP.shape[0] == 0):
            allBvalsGP = np.vstack((bVals))
            # print 'allb shape{}'.format(allBvals.shape)
        else:
            allBvalsGP = np.vstack((allBvalsGP, bVals))

    timesGP.append(time_taken)

    print allBvals.shape

    mu = np.mean(allBvals, axis=0)

    sd = np.std(allBvals, axis=0)

    muGP = np.mean(allBvalsGP, axis=0)

    sdGP = np.std(allBvalsGP, axis=0)

    print 'times mean {}'.format(np.mean(times))
    print 'times sd {}'.format(np.std(times))

    plt.figure()

    x = range(1, (mu.shape[0]) + 1)
    xGP = np.arange(1, (mu.shape[0]) + 1) + 0.1
    plt.errorbar(x, mu, yerr=sd, fmt="o-", label='BNN')
    plt.errorbar(xGP, muGP, yerr=sdGP, fmt="o-", label='GP')

    pylab.grid(True)
    plt.xlabel('Function Evaluation')
    plt.ylabel('Best Value')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    readPickle()
