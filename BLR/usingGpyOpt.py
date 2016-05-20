# coding=utf-8
import GPy
import GPyOpt
import numpy as np
import cPickle as pickle

import time
import objectives
import objectiveGpyOpt

import newFitPlots

def make_bestVals(b):
    bestVals = []

    cur_min = 10000
    for i in b:
        if i < cur_min:
            cur_min = i

        bestVals.append(cur_min)

    return np.asarray(bestVals)


def run_opt(seed):
    f_true = GPyOpt.fmodels.experiments1d.forrester()  # true function
    f_sim = GPyOpt.fmodels.experiments1d.forrester(sd=.1)  # noisy version
    bounds = [(0, 1)]  # problem constrains

    np.random.seed(seed)
    myBopt = GPyOpt.methods.BayesianOptimization(f=f_sim.f,  # function to optimize
                                                 bounds=bounds,  # box-constrains of the problem
                                                 acquisition='EI',  # Selects the Expected improvement
                                                 acquisition_par=0)  # psi parameter is set to zero

    # Run the optimization
    max_iter = 17  # evaluation budget

    myBopt.run_optimization(max_iter,  # Number of iterations
                            acqu_optimize_method='fast_brute',  # method to optimize the acq. function
                            acqu_optimize_restarts=30,  # number of local optimizers
                            eps=10e-8)  # secondary stop criteria
    # myBopt.plot_convergence()

    a, b = myBopt.get_evaluations()

    bestVals = make_bestVals(b)

    return bestVals


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
    myBopt.plot_convergence()

    myBopt.plot_acquisition()
    a, b = myBopt.get_evaluations()

    bestVals = make_bestVals(b)

    return bestVals


def iter_GpyOpt(func, bounds, N_iter, f_true):
    iterBopt = GPyOpt.methods.BayesianOptimization(f=func,  # function to optimize
                                                   bounds=bounds,  # box-constrains of the problem
                                                   acquisition='LCB',  # Selects the Expected improvement
                                                   acquisition_par=100)  # psi parameter is set to zero
    ntest = 500
    xtest = np.linspace(bounds[0][0], bounds[0][1], ntest)
    xtest = xtest.reshape(ntest, 1)
    ytest = f_true.f(xtest)

    for i in range(N_iter):
        np.random.seed(1000)

        iterBopt.run_optimization(max_iter=1,
                                  acqu_optimize_method='fast_random',  # method to optimize the acquisition function
                                  acqu_optimize_restarts=30,  # number of local optimizers
                                  eps=10e-6)  # secondary stop criteria

        iterBopt.plot_acquisition('fit_images_forrester/gpyiteration%.03i.png' % (i + 1))
        filename = 'fit_images_forrester/v2gpyForresteriteration%.03i.png' % (i + 1)
        newFitPlots.plot_acquisitionV2(iterBopt.bounds, iterBopt.input_dim, iterBopt.model, iterBopt.X,
                                       iterBopt.Y, iterBopt.acquisition_func.acquisition_function,
                                       iterBopt.suggested_sample, xtest, ytest, filename)

if __name__ == '__main__':
    # func_name = 'GpyOPTforrester'
    #
    # for seed in range(1000, 1050):
    #     print 'SEED {}'.format(seed)
    #     t0 = time.time()
    #
    #     bVals = run_opt(seed)
    #
    #     t1 = time.time()
    #     time_taken = t1 - t0
    #
    #     toDump = {'bVals': bVals, 't': time_taken, 'seed': seed}
    #     nameOfFile = 'pickles/seed{}BayesOptLogs{}.pkl'.format(seed, func_name)
    #
    #     pickle.dump(toDump, open(nameOfFile, "wb"))
    #     print bVals
    #     print "execution took {} s".format(t1 - t0)
    #
    # t0 = time.time()
    # run_opt(1234)
    # t1 = time.time()
    #
    # print 't taken {}'.format(t1 - t0)

    f_sim = GPyOpt.fmodels.experiments1d.forrester(sd=.1)
    f_true = GPyOpt.fmodels.experiments1d.forrester()
    func = f_sim.f
    func_name = 'GPyOPTForrester'
    bounds = [(0, 1)]
    max_iter = 17

    # f_sim = GPyOpt.fmodels.experiments2d.sixhumpcamel(sd=.1)
    # func_name= 'GPyOPTcamel'
    # bounds = [(-2, 2),(-2,2)]
    # max_iter=34

    # f_sim = GPyOpt.fmodels.experiments2d.mccormick(sd=.1)
    # func=f_sim.f
    # func_name= 'GPyOPTmccormick'
    # bounds = [(-1.5, 4),(-3,4)]
    # max_iter=34

    # # f_sim = objectiveGpyOpt.rosenbrock2D(sd=0.1)
    # func = objectives.rosenbrock_2D_noisy
    # func_name='GPyOPT{}LCB'.format(func.func_name)
    # bounds = [(-2, 2), (-2, 2)]
    # max_iter = 34

    # for seed in range(1000, 1001):
    #     print 'SEED {}'.format(seed)
    #     print   func.func_name
    #     t0 = time.time()
    #
    #     bVals = run_optV2(seed,func,bounds,max_iter)
    #
    #     t1 = time.time()
    #     time_taken = t1 - t0
    #
    #     toDump = {'bVals': bVals, 't': time_taken, 'seed': seed}
    #     nameOfFile = 'pickles1/seed{}BayesOptLogs{}.pkl'.format(seed, func_name) #TODO
    #
    #     # pickle.dump(toDump, open(nameOfFile, "wb")) #TODO remove to store pickles
    #
    #     print bVals
    #     print bVals.shape
    #     print "execution took {} s".format(t1 - t0)



    iter_GpyOpt(func, bounds, max_iter, f_true=f_true)















    # t0 = time.time()
    # run_opt(1234)
    # t1 = time.time()
    #
    # print 't taken {}'.format(t1 - t0)
