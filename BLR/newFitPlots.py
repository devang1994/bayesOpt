# coding=utf-8


import numpy as np
from pylab import grid
import matplotlib.pyplot as plt
from pylab import savefig


def plot_acquisition(bounds, input_dim, model, Xdata, Ydata, acquisition_function, suggested_sample, filename=None):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    '''

    # Plots in dimension 1
    if input_dim == 1:
        X = np.arange(bounds[0][0], bounds[0][1], 0.001)
        X = X.reshape(len(X), 1)
        acqu = acquisition_function(X)
        acqu_normalized = (-acqu - min(-acqu)) / (max(-acqu - min(-acqu)))  # normalize acquisition
        m, v = model.predict(X.reshape(len(X), 1))
        plt.ioff()
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(X, m, 'b-', label=u'Posterior mean', lw=2)
        plt.fill(np.concatenate([X, X[::-1]]), \
                 np.concatenate([m - 1.9600 * np.sqrt(v),
                                 (m + 1.9600 * np.sqrt(v))[::-1]]), \
                 alpha=.5, fc='b', ec='None', label='95% C. I.')
        plt.plot(X, m - 1.96 * np.sqrt(v), 'b-', alpha=0.5)
        plt.plot(X, m + 1.96 * np.sqrt(v), 'b-', alpha=0.5)
        plt.plot(Xdata, Ydata, 'r.', markersize=10, label=u'Observations')
        plt.axvline(x=suggested_sample[len(suggested_sample) - 1], color='r')
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


def plot_acquisitionV2(bounds, input_dim, model, Xdata, Ydata, acquisition_function, suggested_sample, xtest, ytest,
                       filename=None):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    '''

    X = np.arange(bounds[0][0], bounds[0][1], 0.001)
    X = X.reshape(len(X), 1)
    acqu = acquisition_function(X)
    acqu_normalized = (-acqu - min(-acqu)) / (max(-acqu - min(-acqu)))  # normalize acquisition
    m, v = model.predict(X.reshape(len(X), 1))
    plt.ioff()
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(X, m, 'b-', label=u'Posterior mean', lw=2)
    plt.fill(np.concatenate([X, X[::-1]]), \
             np.concatenate([m - 1.9600 * np.sqrt(v),
                             (m + 1.9600 * np.sqrt(v))[::-1]]), \
             alpha=.5, fc='b', ec='None', label='95% C. I.')
    plt.plot(X, m - 1.96 * np.sqrt(v), 'b-', alpha=0.5)
    plt.plot(X, m + 1.96 * np.sqrt(v), 'b-', alpha=0.5)
    plt.plot(Xdata, Ydata, 'r.', markersize=10, label=u'Observations')
    plt.axvline(x=suggested_sample[len(suggested_sample) - 1], color='r')
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

    plt.figure(figsize=(10, 6))

    m, v = model.predict(xtest)
    mu = m
    sd = np.sqrt(v)

    plt.plot(xtest, ytest, color='black', label='objective', linewidth=2.0)
    plt.plot(Xdata, Ydata, 'ro')
    plt.plot(xtest, mu, color='r', label='posterior')
    plt.fill(np.concatenate([xtest, xtest[::-1]]),
             np.concatenate([mu - 1.9600 * sd,
                             (mu + 1.9600 * sd)[::-1]]),
             alpha=.3, fc='b', ec='None', label='95% C. I.')
    plt.legend(loc='best')

    if filename != None:
        savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()
