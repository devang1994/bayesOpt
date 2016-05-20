# coding=utf-8
import numpy as np
from pylab import grid
import matplotlib.pyplot as plt
from pylab import savefig

import numpy as np
import bayesOptBNN
import objectives
import matplotlib.pyplot as plt

import newFitPlots

import cPickle as pickle


def plot_acquisition(X, m, sd, Xdata, Ydata, acqu, filename=None):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    acqu is acquisition function at test points

    '''

    # Plots in dimension 1

    # X = np.arange(bounds[0][0], bounds[0][1], 0.001)
    # X = X.reshape(len(X),1)
    # acqu = acquisition_function(X)
    acqu_normalized = (-acqu - min(-acqu)) / (max(-acqu - min(-acqu)))  # normalize acquisition
    # m, v = model.predict(X.reshape(len(X),1))
    plt.ioff()
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(X, m, 'b-', label=u'Posterior mean', lw=2)
    plt.fill(np.concatenate([X, X[::-1]]), \
             np.concatenate([m - 1.9600 * sd,
                             (m + 1.9600 * sd)[::-1]]), \
             alpha=.5, fc='b', ec='None', label='95% C. I.')
    plt.plot(X, m - 1.96 * sd, 'b-', alpha=0.5)
    plt.plot(X, m + 1.96 * sd, 'b-', alpha=0.5)
    plt.plot(Xdata, Ydata, 'r.', markersize=10, label=u'Observations')
    # plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
    plt.title('Model and observations')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.legend(loc='upper left')
    plt.xlim(*bounds)
    grid(True)
    plt.subplot(2, 1, 2)
    plt.axvline(x=suggested_sample[len(suggested_sample) - 1], color='r')
    plt.plot(X, acqu_normalized, 'r-', lw=2)
    plt.xlabel('X')
    plt.ylabel('Acquisition value')
    plt.title('Acquisition function')
    grid(True)
    plt.xlim(*bounds)
    if filename != None:
        savefig(filename)
    else:
        plt.show()


def read_pickle_makePlots():
    func = objectives.objectiveForrester
    i = 0
    func_name = 'objectiveForrester'
    seed = 1000
    # bounds=[(0,1)]
    input_dim = 1
    xr = [0, 1]

    # print 'pickling'
    # interm = {'mu': mu, 'sd': sd}
    nameOfFile1 = 'pickles_evo/it{}seed{}BayesOptEvo{}.pkl'.format(i, seed, func_name)
    print 'un pickling {}'.format(nameOfFile1)
    dumpFile = pickle.load(open(nameOfFile1, "rb"))

    mu = dumpFile['mu']
    sd = dumpFile['sd']
    ntest = 500

    xtest = np.linspace(xr[0], xr[1], ntest)
    xtest = xtest.reshape(ntest, 1)

    ytest = func(xtest)

    plot_acquisition(xtest, input_dim, mu, sd, iterBopt.X,
                     iterBopt.Y, iterBopt.acquisition_func.acquisition_function, iterBopt.suggested_sample,
                     filename)


if __name__ == '__main__':
    # func=objectives.objectiveForrester
    # xr=[0,1]
    # actual_min=-6.02074
    # numDim = 1
    # init_random = 2
    # k = 10
    # # num_it=18
    # num_it=18
    # numDim = len(xr) / 2

    # func = objectives.rosenbrock_2D
    # xr = [-2, 2, -2, 2]  # generalized to multiD (2d)
    # actual_min = 0
    # numDim = 2
    # init_random = 5
    # k = 10
    # num_it = 35
    # numDim = len(xr) / 2
    #
    # seed=1000
    # bVals = bayesOptBNN.bayes_opt(func, xr, initial_random=init_random, num_it=num_it, k=k, hWidths=[50, 50, 50],
    #                   precisions=[1, 1, 1, 1], vy=100,
    #                   show_evo=False, actual_min=actual_min, numDim=numDim, seed=seed,pickle_evo=True)
    #
    # func = objectives.sixhumpcamel
    # xr = [-2, 2, -2, 2]  # generalized to multiD (2d)
    # actual_min = -1.0316
    # init_random = 5
    # k = 10
    # num_it = 35
    # numDim = len(xr) / 2
    #
    # seed=1000
    # bVals = bayesOptBNN.bayes_opt(func, xr, initial_random=init_random, num_it=num_it, k=k, hWidths=[50, 50, 50],
    #                   precisions=[1, 1, 1, 1], vy=100,
    #                   show_evo=False, actual_min=actual_min, numDim=numDim, seed=seed,pickle_evo=True)
    #
    #
    # func = objectives.mccormick
    # xr = [-1.5, 4, -3, 4]  # generalized to multiD (2d)
    # actual_min = -1.9133
    # init_random = 5
    # k = 10
    # num_it = 35
    # numDim = len(xr) / 2
    #
    #
    # seed=1000
    # bVals = bayesOptBNN.bayes_opt(func, xr, initial_random=init_random, num_it=num_it, k=k, hWidths=[50, 50, 50],
    #                   precisions=[1, 1, 1, 1], vy=100,
    #                   show_evo=False, actual_min=actual_min, numDim=numDim, seed=seed,pickle_evo=True)


    # plt.show()

    read_pickle_makePlots()
