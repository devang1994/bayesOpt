# coding=utf-8
import GPy
import GPyOpt
import numpy as np
import cPickle as pickle

import time
import objectives
import objectiveGpyOpt

import newFitPlots
import matplotlib.pyplot as plt


def run_optV2(seed, func, bounds, max_iter):
    # f_true = GPyOpt.fmodels.experiments1d.forrester()  # true function
    # f_sim = GPyOpt.fmodels.experiments1d.forrester(sd=.1)  # noisy version
    # bounds = [(0, 1)]  # problem constrains
    print func.func_name
    np.random.seed(seed)
    # myBopt = GPyOpt.methods.BayesianOptimization(f=func,  # function to optimize
    #                                              bounds=bounds,  # box-constrains of the problem
    #                                              acquisition='EI',  # Selects the Expected improvement
    #                                              acquisition_par=0)  # psi parameter is set to zero


    myBopt = GPyOpt.methods.BayesianOptimization(f=func,  # function to optimize
                                                 bounds=bounds,  # box-constrains of the problem
                                                 acquisition='LCB',  # Selects the Expected improvement
                                                 acquisition_par=100)  # psi parameter is set to zero

    # Run the optimization
    # max_iter = 17  # evaluation budget

    myBopt.run_optimization(max_iter,  # Number of iterations
                            acqu_optimize_method='fast_brute',  # method to optimize the acq. function
                            acqu_optimize_restarts=30,  # number of local optimizers
                            eps=10e-8)  # secondary stop criteria
    # myBopt.plot_convergence()

    myBopt.plot_acquisition()
    plt.show()
    a, b = myBopt.get_evaluations()

    # bestVals = make_bestVals(b)
    #
    # return bestVals


if __name__ == '__main__':

    # # f_sim = objectiveGpyOpt.rosenbrock2D(sd=0.1)
    # func = objectives.rosenbrock_2D_noisy
    # func_name='GPyOPT{}LCB'.format(func.func_name)
    # bounds = [(-2, 2), (-2, 2)]
    # max_iter = 34

    f_sim = GPyOpt.fmodels.experiments2d.sixhumpcamel(sd=.1)
    func=f_sim.f
    func_name= 'GPyOPTcamel'
    bounds = [(-2, 2),(-2,2)]
    max_iter=34


    run_optV2(1000,func,bounds,max_iter)