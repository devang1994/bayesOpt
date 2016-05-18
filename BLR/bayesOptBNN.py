# coding=utf-8

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
np.random.seed(42)


def produce_mu_and_sd(n_samples, hWidths, xtrain, ytrain, xtest, ytest, precisions, vy, burnin=0, seed=12345):
    train_err, test_err, samples, train_op_samples = sampler_on_BayesNN(burnin=0, n_samples=n_samples,
                                                                        precisions=precisions,
                                                                        vy=vy,
                                                                        X_train=xtrain, y_train=ytrain,
                                                                        hWidths=hWidths,
                                                                        stepsize=0.001,
                                                                        n_steps=30, seed=seed)
    # print 'sampling worked'

    ntrain = xtrain.shape[0]
    test_pred, test_sd = analyse_samples(samples, xtrain, ytrain, hWidths=hWidths, burnin=burnin, display=False,
                                         title='ntrain {}'.format(ntrain), X_test=xtest, y_test=ytest)

    return test_pred, test_sd


def bayes_opt(func, xr, hWidths, precisions, vy, numDim, actual_min=0.0, initial_random=2, k=0.2, num_it=20,
              show_evo=False, seed=12345):
    '''function to do bayesOpt on and number of initial random evals
    noise is artificially added to objective function calls when training
    '''


    print 'init_rand {}, k {}, num_it {}, func {}'.format(initial_random, k, num_it, func.func_name)

    noise_var = 0.01
    ntest = 500
    ntrain = initial_random  # number of initial random function evals
    if (numDim == 2):
        x1 = np.random.uniform(low=xr[0], high=xr[1], size=(ntrain, 1))
        x2 = np.random.uniform(low=xr[2], high=xr[3], size=(ntrain, 1))
        xtrain = np.hstack((x1, x2))  # shape (ntrain,2)
    elif (numDim == 1):
        xtrain = np.random.uniform(low=xr[0], high=xr[1], size=(ntrain, 1))
    else:
        print 'Wrong number of dimensions, not yet implemented'
        return

    input_size = xtrain.shape[1]
    # xtrain = np.random.uniform(low=xr[0], high=xr[1], size=(ntrain, 1))
    # print xtrain.shape
    ytrain = func(xtrain) + np.random.randn(ntrain, 1) * sqrt(noise_var)
    # print ytrain.shape
    # print ytrain
    if (numDim == 2):
        x1 = np.linspace(xr[0], xr[1], ntest)
        x1 = x1.reshape(ntest, 1)

        x2 = np.linspace(xr[2], xr[3], ntest)
        x2 = x2.reshape(ntest, 1)

        xtest = np.hstack((x1, x2))
    elif (numDim == 1):
        xtest = np.linspace(xr[0], xr[1], ntest)
        xtest = xtest.reshape(ntest, 1)

    else:
        print 'Wrong number of dimensions, not yet implemented'
        return



    ytest = func(xtest)

    # print xtest.shape
    # print ytest.shape
    ytrain_pure = func(xtrain)
    cur_min_index = np.argmin(ytrain_pure)

    cur_miny = ytrain_pure[cur_min_index]
    # print type(cur_miny)
    cur_minx = xtrain[cur_min_index]

    print 'original min {}'.format(cur_miny)

    best_vals = []
    best_vals.extend((np.ones(initial_random)) * cur_miny)

    for i in range(num_it):
        print 'it:{}'.format(i)

        mu, sd = produce_mu_and_sd(n_samples=1000, hWidths=hWidths, xtrain=xtrain, ytrain=ytrain,
                                   precisions=precisions, vy=vy, burnin=100, xtest=xtest, ytest=ytest, seed=seed)

        alpha = acquisition_UCB(mu, sd, k=k)

        index = np.argmin(alpha)

        next_query = xtest[index, :]
        next_query = next_query.reshape(1, input_size)
        # print 'index {}, nextq {} '.format(index, next_query)


        next_y = func(next_query) + np.random.randn(1, 1) * sqrt(noise_var)
        next_y_pure = func(next_query)

        if (next_y_pure < cur_miny):
            cur_miny = next_y_pure
            cur_minx = next_query
        print 'query pt {}'.format(next_query)
        print 'cur miny {} , cur_y {} '.format(cur_miny, next_y_pure)
        # print 'nexty pure.shape {}, cur_miny.shape {}'.format(next_y_pure.shape,cur_miny.shape)

        best_vals.append(cur_miny)
        s = sd  # standard deviations

        if (i % 4 == 0 and show_evo):
            # plt.figure()
            f, axarr = plt.subplots(2, sharex=True)

            # .scatter(x, y)
            axarr[1].plot(xtest, func(xtest), color='black', label='objective', linewidth=2.0)
            axarr[1].plot(xtrain, ytrain, 'ro')
            axarr[1].plot(xtest, mu, color='r', label='posterior')
            # plt.plot(xtest, mu - s, color='blue', label='credible')
            # plt.plot(xtest, mu + s, color='blue', label='interval')
            axarr[1].plot(xtest, alpha, label='acquistion func', color='green')
            # plt.plot(xtest,np.zeros(ntest),color='black')
            axarr[0].plot(xtest, s, label='sigma', color='blue')
            axarr[0].set_title('BNN with hMC,ntrain:{}'.format(xtrain.shape[0]))
            plt.legend(fontsize='x-small')
            # plt.draw()
            # plt.savefig('bayesOptNtrain{}k{}init{}.png'.format(xtrain.shape[0], k, ntrain), dpi=300)
            # plt.show(block=False)
        xtrain = np.vstack((xtrain, next_query))
        ytrain = np.vstack((ytrain, next_y))

    plt.figure()
    plt.plot(best_vals, '-o')
    plt.xlabel('Function Evalutation')
    plt.ylabel('Best Value')
    plt.title('{}'.format(func.func_name))
    pylab.grid(True)

    plt.figure()
    best_vals = np.asarray(best_vals)
    plt.plot(np.abs(best_vals - actual_min), '-o')
    plt.ylabel('Optimality Gap')
    plt.xlabel('Function Evalutation')
    plt.title('{}'.format(func.func_name))
    pylab.grid(True)
    return best_vals


if __name__ == '__main__':
    # func = objectives.objectiveGramacyLee
    # xr = [0.5, 2.5]
    # numDim = 1

    # func=objectives.objectiveForrester
    # xr=[0,1]
    # actual_min=-6.02074
    # numDim = 1

    # func = objectives.brannin_hoo
    # xr = [-5, 10, 0, 15]  # generalized to multiD (2d)
    # actual_min = 0.397887
    # numDim = 2

    # func = objectives.rosenbrock_2D
    # xr = [-2, 2, -2, 2]  # generalized to multiD (2d)
    # actual_min = 0
    # numDim = 2

    # func = objectives.modified_rescaled_brannin_hoo
    # xr = [0, 1, 0, 1]  # generalized to multiD (2d)
    # actual_min = -0.5214


    func = objectives.objectiveSinCos
    xr = [0, 1]  # generalized to multiD (2d)
    actual_min = -1.96729
    numDim = 1
    init_random = 2
    k = 10

    numDim = len(xr) / 2
    t0 = time.time()

    # print 'lower minstepsize brannin with 30, evo, k=10 '
    bVals = bayes_opt(func, xr, initial_random=init_random, num_it=5, k=k, hWidths=[50, 50, 50],
                      precisions=[1, 1, 1, 1], vy=100,
                      show_evo=True, actual_min=actual_min, numDim=numDim, seed=12345)

    t1 = time.time()
    print "execution took {} s".format(t0 - t1)
    plt.show()
