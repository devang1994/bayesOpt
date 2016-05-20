# coding=utf-8


try:
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
except:
    pass
import numpy as np


def reshape(x, input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size == input_dim:
        x = x.reshape((1, input_dim))
    return x


class function2d:
    '''
    This is a benchmark of bidimensional functions interesting to optimize.

    '''

    def plot(self):
        bounds = self.bounds
        x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
        x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.hstack((X1.reshape(100 * 100, 1), X2.reshape(100 * 100, 1)))
        Y = self.f(X)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X1, X2, Y.reshape((100, 100)), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0,
                        antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_title(self.name)

        plt.figure()
        plt.contourf(X1, X2, Y.reshape((100, 100)), 100)
        if (len(self.min) > 1):
            plt.plot(np.array(self.min)[:, 0], np.array(self.min)[:, 1], 'w.', markersize=20, label=u'Observations')
        else:
            plt.plot(self.min[0][0], self.min[0][1], 'w.', markersize=20, label=u'Observations')
        plt.colorbar()
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(self.name)
        plt.show()


class rosenbrock2D(function2d):
    '''
    Rosenbrock function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds == None:
            self.bounds = [(-2, 2), (-2, 2)]
        else:
            self.bounds = bounds
        self.min = [(1., 1.)]
        self.fmin = 0.0
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'Rosenbrock'

    def f(self, x):
        x = reshape(x, self.input_dim)
        n = x.shape[0]
        if x.shape[1] != self.input_dim:
            return 'wrong input dimension'
        else:
            x1 = x[:, 0]
            x2 = x[:, 1]
            a = 1.0
            b = 100.0
            term1 = (a - x1) ** 2
            term2 = b * ((x2 - (x1 ** 2)) ** 2)

            fval = term1 + term2
            if self.sd == 0:
                noise = np.zeros(n).reshape(n, 1)
            else:
                noise = np.random.normal(0, self.sd, n).reshape(n, 1)
            return fval.reshape(n, 1) + noise
