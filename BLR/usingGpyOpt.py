# coding=utf-8
import GPy
import GPyOpt
import numpy as np
import cPickle as pickle

import time


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


if __name__ == '__main__':

    func_name = 'GpyOPTforrester'

    for seed in range(1000, 1050):
        print 'SEED {}'.format(seed)
        t0 = time.time()

        bVals = run_opt(seed)

        t1 = time.time()
        time_taken = t1 - t0

        toDump = {'bVals': bVals, 't': time_taken, 'seed': seed}
        nameOfFile = 'pickles/seed{}BayesOptLogs{}.pkl'.format(seed, func_name)

        pickle.dump(toDump, open(nameOfFile, "wb"))
        print bVals
        print "execution took {} s".format(t1 - t0)

    t0 = time.time()
    run_opt(1234)
    t1 = time.time()

    print 't taken {}'.format(t1 - t0)
