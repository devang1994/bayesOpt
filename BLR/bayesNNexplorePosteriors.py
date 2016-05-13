from bayesNN_HMCv2 import sampler_on_BayesNN, objective, analyse_samples
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import cPickle as pickle
from find_effective_sampleSize import RCodaTools
from MAPnet import mlp_synthetic


def mixing(sf, vy, show_fit=False, show_post=False):
    '''

    :param sf: scale factor for precisions
    :param vy: precision of noise
    :return:
    '''
    ntrain = 20
    noise_var = 0.01
    X_train = np.random.uniform(low=-1.0, high=1.0, size=ntrain).reshape(ntrain, 1)
    # print X_train.shape
    y_train = objective(X_train) + np.random.randn(ntrain, 1) * sqrt(noise_var)

    ntest = 1000
    X_test = np.linspace(-1., 1., ntest)
    y_test = objective(X_test)
    X_test = X_test.reshape(ntest, 1)
    y_test = y_test.reshape(ntest, 1)


    # precisions = [0.6125875773048164, 0.03713439386866191, 14.22759780450891, 5.72501724650353]
    # vy = 4.631095917555727

    precisions = [1, 1, 1, 1]
    # precisions = [1, 1]

    precisions = [sf * x for x in precisions]
    # hWidths = [50, 50, 50]

    hWidths = [50, 50, 50]

    # a, b, init_MAP = mlp_synthetic(X_train, X_test, y_train, y_test, precision=precisions[0], vy=vy, hWidths=hWidths,
    #                                display=True, epochs=4000)
    # # plt.show()
    # print 'finished MAP'
    # analyse_samples(init_MAP,X_train, y_train, hWidths=hWidths, burnin=0, display=True,title='MAP')
    # plt.show()
    train_err, test_err, samples, train_op_samples = sampler_on_BayesNN(burnin=0, n_samples=1000, precisions=precisions,
                                                                        vy=vy,
                                                                        X_train=X_train, y_train=y_train,
                                                                        hWidths=hWidths, target_acceptance_rate=0.9,
                                                                        stepsize=0.001,
                                                                        n_steps=30)

    # print RCodaTools.ess_coda_vec(np.transpose(samples))
    # , init_theta=init_MAP

    print 'effective sample sizes'
    a = RCodaTools.ess_coda_vec(samples)
    print np.mean(a)
    print np.min(a)

    # w1 = samples[:, 1]
    # w2 = samples[:, 5200]
    # w3 = samples[:, 1200]
    # w4 = samples[:, 200]

    w1 = samples[:, 1]
    w2 = samples[:, 200]
    w3 = samples[:, 251]
    # w4 = samples[:, 200]

    plt.figure()
    plt.plot(w1, label='w1')
    plt.plot(w2, label='w2')
    plt.plot(w3, label='w3')
    plt.title('weight prec {}, noise prec {}'.format(sf, vy))
    plt.legend()

    plt.xlabel('Num Iterations')
    plt.ylabel('Value')

    # plt.savefig('logs/BNN_logs/mixingWeightsPrec10L', dpi=300)

    print samples.shape

    analyse_samples(samples, X_train, y_train, hWidths=hWidths, burnin=200, display=True, title='Averaged')

    # analyse_samples((samples[1750,:]).reshape(1,-1),X_train, y_train, hWidths=hWidths, burnin=0, display=True,title='sample=1750')

    analyse_samples((samples[240, :]).reshape(1, -1), X_train, y_train, hWidths=hWidths, burnin=0, display=True,
                    title='sample=240')
    # analyse_samples((samples[4000, :]).reshape(1,-1), X_train, y_train, hWidths=hWidths, burnin=0, display=True,title='sample=4000')

    if (show_post):
        samples = samples[200:, :]  # burning in

        w1 = samples[:, 1]
        w2 = samples[:, 5200]
        w3 = samples[:, 1200]
        plt.figure()

        N = samples.shape[0]
        n = N / 10

        plt.hist(w1, bins=n)  # bin it into n = N/10 bins
        plt.figure()

        plt.hist(w2, bins=n)  # bin it into n = N/10 bins
        plt.figure()

        plt.hist(w3, bins=n)  # bin it into n = N/10 bins


def mixing_from_pickle():
    burnin = 200
    samples = pickle.load(open("logs/BNN_logs/samples_gibbs_acc0.6_2000_sh10.p", "rb"))
    w1 = samples[:, 1]
    w2 = samples[:, 5200]
    w3 = samples[:, 1200]
    w4 = samples[:, 200]

    plt.plot(w1, label='w1')
    plt.plot(w2, label='w2')
    plt.plot(w3, label='w3')
    plt.legend()

    plt.xlabel('Num Iterations')
    plt.ylabel('Value')

    # plt.savefig('logs/BNN_logs/mixingWeightsPrec10L', dpi=300)

    print samples.shape

    samples = samples[burnin:, :]  # burning in

    w1 = samples[:, 1]
    w2 = samples[:, 5200]
    w3 = samples[:, 1200]
    plt.figure()

    N = samples.shape[0]
    n = N / 10

    plt.hist(w1, bins=n)  # bin it into n = N/10 bins
    # x = x[:-1] + (x[1] - x[0]) / 2  # convert bin edges to centers
    # f = UnivariateSpline(x, p, s=n)
    # plt.plot(x, f(x))
    plt.figure()

    plt.hist(w2, bins=n)  # bin it into n = N/10 bins
    # x = x[:-1] + (x[1] - x[0]) / 2  # convert bin edges to centers
    # f = UnivariateSpline(x, p, s=n)
    # plt.plot(x, f(x))
    plt.figure()

    plt.hist(w3, bins=n)  # bin it into n = N/10 bins
    # x = x[:-1] + (x[1] - x[0]) / 2  # convert bin edges to centers
    # f = UnivariateSpline(x, p, s=n)
    # plt.plot(x, f(x))

    plt.show()


# mixing()



if __name__ == '__main__':
    # mixing_from_pickle()
    # sfVals=[1,10]
    # vVals=[1,10]
    # for sf in sfVals:
    #     for vy in vVals:
    #         mixing(sf,vy)
    #
    # mixing(1,1,show_fit=True)
    # mixing(1,10,show_fit=True)
    mixing(1, 100, show_fit=True)
    # mixing(10,1,show_fit=True)
    plt.show()

"""
Summary

sf 1 , vy 100 . no gibbs

try out BayesOpt with this

To confirm that this multi mode mixing thing is correct, pull out a few samples from different parts in the trace
and show that they have the same fit
"""
