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
    # func_name = 'objectiveForrester'
    # func_name_GP = 'GpyOPTforrester'
    # totalFevals = 20

    # seed1000BayesOptLogsGPyOPTcamel
    # seed1008BayesOptLogssixhumpcamel
    # func_name = 'sixhumpcamel'
    # func_name_GP = 'GPyOPTcamel'
    # totalFevals = 40

    # seed1000BayesOptLogsGPyOPTmccormick
    # seed1045BayesOptLogsmccormick

    # func_name = 'mccormick'
    # func_name_GP = 'GPyOPTmccormick'
    # totalFevals = 40

    # seed1000BayesOptLogsrosenbrock_2D
    # seed1000BayesOptLogsGPyOPTRosenbrock2D
    # seed1049BayesOptLogsGPyOPTrosenbrock_2D_noisy
    # func_name = 'rosenbrock_2D'
    # func_name_GP = 'GPyOPTrosenbrock_2D_noisy'
    # totalFevals = 40


    # k35seed1049init_random6BayesOptLogssixhumpcamel
    # k40seed1049init_random6BayesOptLogssixhumpcamel

    func_name='sixhumpcamel'

    prefix='k40'
    mid='init_random6'

    func_name_GP = 'GPyOPTcamel'
    totalFevals = 40


    # actual_min = -1.96729
    allBvals = np.asarray([])
    times = []

    allBvalsGP = np.asarray([])
    timesGP = []

    for seed in range(1000, 1050):
        print seed
        # seed=1000
        # toDump = {'bVals': bVals, 't': time_taken, 'seed': seed, 'k': k, 'init_random': init_random}
        nameOfFile = 'pickles/{}seed{}{}BayesOptLogs{}.pkl'.format(prefix,seed,mid, func_name)

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
            bVals = np.hstack((bVals, temp))
            print bVals

        if (allBvalsGP.shape[0] == 0):
            allBvalsGP = np.vstack((bVals))
            # print 'allb shape{}'.format(allBvals.shape)
        else:
            allBvalsGP = np.vstack((allBvalsGP, bVals))

    timesGP.append(time_taken)

    print allBvals.shape

    # mu = np.mean(allBvals, axis=0)
    #
    # sd =0.5* np.std(allBvals, axis=0)
    #
    # muGP = np.mean(allBvalsGP, axis=0)
    #
    # sdGP = 0.5*np.std(allBvalsGP, axis=0)

    mu=np.median(allBvals, axis=0)

    print mu
    lower=mu-(np.percentile(allBvals,25, axis=0))
    upper=np.percentile(allBvals,75, axis=0)-mu
    sd=[lower, upper]
    muGP = np.median(allBvalsGP, axis=0)

    lowerGP =muGP- np.percentile(allBvalsGP, 25, axis=0)
    upperGP = np.percentile(allBvalsGP, 75, axis=0)-muGP
    sdGP=[lowerGP,upperGP]





    print muGP

    print 'times mean {}'.format(np.mean(times))
    print 'times sd {}'.format(np.std(times))

    plt.figure()

    x = range(1, (mu.shape[0]) + 1)
    xGP = np.arange(1, (mu.shape[0]) + 1) + 0.15
    plt.errorbar(x, mu, yerr=sd, fmt="o-", label='BNN')
    xGP=xGP[0:40]
    print xGP.shape
    print muGP.shape
    print sdGP[0].shape
    plt.errorbar(xGP[0:40], muGP, yerr=sdGP, fmt="o-", label='GP')

    pylab.grid(True)
    plt.xlabel('Function Evaluation')
    plt.ylabel('Best Value')
    plt.legend(loc='best')
    # plt.savefig('report_images/{}BestValsBoxLike.png'.format(func_name), dpi=300, bbox_inches='tight')


    # plt.figure()
    # bp=plt.boxplot(allBvals,0,'',patch_artist=True)


    # bp1=plt.boxplot(allBvalsGP,0,'',patch_artist=True)


    ## add patch_artist=True option to ax.boxplot()
    ## to get fill color
    # bp = ax.boxplot(data_to_plot, patch_artist=True)

    ## change outline color, fill color and linewidth of the boxes
    # for box in bp['boxes']:
    #     # change outline color
    #     box.set(color='#7570b3', linewidth=2,alpha=0.0)
    #     # change fill color
    #     box.set(facecolor='#1b9e77')
    #
    # ## change color and linewidth of the whiskers
    # for whisker in bp['whiskers']:
    #     whisker.set(color='#7570b3', linewidth=2)
    #
    # ## change color and linewidth of the caps
    # for cap in bp['caps']:
    #     cap.set(color='#7570b3', linewidth=2)
    #
    # ## change color and linewidth of the medians
    # for median in bp['medians']:
    #     median.set(color='#b2df8a', linewidth=2)
    #
    # ## change the style of fliers and their fill
    # for flier in bp['fliers']:
    #     flier.set(marker='o', color='#e7298a', alpha=0.0)

    pylab.grid(True)
    plt.show()


if __name__ == '__main__':
    readPickle()
