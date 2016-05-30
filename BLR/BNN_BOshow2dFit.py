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


# def plot_acquisition_2D(bounds,input_dim,model,Xdata,Ydata,acquisition_function,suggested_sample, filename = None):

# print 'pickling'
# interm = {'mu': mu, 'sd': sd}
# nameOfFile1 = 'pickles_evo/it{}seed{}BayesOptEvo{}.pkl'.format(i, seed, func.func_name)
# print 'pickling {}'.format(nameOfFile1)

def plot_acquisition_2D(bounds,input_dim,model,Xdata,Ydata,acquisition_function,suggested_sample, filename = None):

    X1 = np.linspace(bounds[0][0], bounds[0][1], 200)
    X2 = np.linspace(bounds[1][0], bounds[1][1], 200)
    x1, x2 = np.meshgrid(X1, X2)
    X = np.hstack((x1.reshape(200 * 200, 1), x2.reshape(200 * 200, 1)))
    acqu = acquisition_function(X)
    acqu_normalized = (-acqu - min(-acqu)) / (max(-acqu - min(-acqu)))
    acqu_normalized = acqu_normalized.reshape((200, 200))
    m, v = model.predict(X)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.contourf(X1, X2, m.reshape(200, 200), 100)
    plt.plot(Xdata[:, 0], Xdata[:, 1], 'r.', markersize=10, label=u'Observations')
    plt.colorbar()
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Posterior mean')
    plt.axis((bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]))
    ##
    plt.subplot(1, 3, 2)
    plt.plot(Xdata[:, 0], Xdata[:, 1], 'r.', markersize=10, label=u'Observations')
    plt.contourf(X1, X2, np.sqrt(v.reshape(200, 200)), 100)
    plt.colorbar()
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Posterior sd.')
    plt.axis((bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]))
    ##
    plt.subplot(1, 3, 3)
    plt.contourf(X1, X2, acqu_normalized, 100)
    plt.colorbar()
    plt.plot(suggested_sample[:, 0], suggested_sample[:, 1], 'k.', markersize=10)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Acquisition function')
    plt.axis((bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]))
    if filename != None: savefig(filename)



# print 'pickling'
# interm = {'mu': mu, 'sd': sd}
# nameOfFile1 = 'pickles_evo/it{}seed{}BayesOptEvo{}.pkl'.format(i, seed, func.func_name)
# print 'pickling {}'.format(nameOfFile1)
def readPickle():


    i=34
    seed=1000
    func = objectives.sixhumpcamel
    nameOfFile1 = 'pickles_evo/it{}seed{}BayesOptEvo{}.pkl'.format(i, seed, func.func_name)

    interm=pickle.load(open(nameOfFile1, "rb"))

    mu=interm['mu']
    sd=interm['sd']

    print mu.shape
    print sd.shape


readPickle()


