# coding=utf-8
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

np.random.seed(42)


def unpack_theta(theta, hWidths, input_size, output_size, index=0):
    # w1 = theta[index, 0:hidden_width * input_size].reshape((input_size, hidden_width))
    # b1 = theta[index, hidden_width * input_size:hidden_width * input_size + hidden_width]
    # wo = theta[index,
    #      hidden_width * input_size + hidden_width:hidden_width * input_size + hidden_width + hidden_width * output_size].reshape(
    #     (hidden_width, -1))
    # temp = hidden_width * input_size + hidden_width + hidden_width * output_size
    # bo = theta[index, temp:temp + output_size]
    #
    # return w1, b1, wo, bo
    weights = []
    biases = []
    widths = hWidths[:]
    widths.insert(0, input_size)
    widths.append(output_size)
    # print widths
    cur = 0

    for i in range(len(widths) - 1):
        w = theta[index, cur:cur + widths[i] * widths[i + 1]].reshape((widths[i], widths[i + 1]))
        cur = cur + widths[i] * widths[i + 1]
        weights.append(w)
        b = theta[index, cur: cur + widths[i + 1]]
        biases.append(b)
        cur = cur + widths[i + 1]

    return weights, biases


def model(X, theta, hWidths, input_size, output_size):
    weights, biases = unpack_theta(theta, hWidths, input_size, output_size)

    h = X
    for i in range(len(weights) - 1):
        w = weights[i]
        b = biases[i]
        h = T.tanh(T.dot(h, w) + b)

    op = T.dot(h, weights[-1]) + biases[-1]
    return op


def floatX(X):
    """convert to np array with floatX"""
    return np.asarray(X, dtype=theano.config.floatX)


def sgd(cost, params, lr=0.005):
    """same as simple update, just does it over all params"""
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates


def uniform_weights(shape):
    scale = sqrt(6. / (shape[1] + shape[0]))
    return theano.shared(floatX(np.random.uniform(low=-scale, high=scale, size=shape)))


def find_dim_theta(hWidths, input_size, output_size):
    dim = np.sum(hWidths) + output_size  # bisaes

    for i in range(len(hWidths) - 1):
        dim = dim + hWidths[i] * hWidths[i + 1]

    dim = dim + hWidths[0] * input_size
    dim = dim + hWidths[-1] * output_size
    return dim
    pass


def mlp_synthetic(X_train, X_test, y_train, y_test, precision, vy, hWidths, mini_batchsize=20, epochs=1000,
                  display=False):
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    X = T.fmatrix(name='X')
    Y = T.fmatrix(name='Y')
    rng = numpy.random.RandomState(123)
    dim = find_dim_theta(hWidths, input_size, output_size)

    input_size = X_train.shape[1]
    initial_params = theano.shared(floatX(rng.randn(1, dim).astype(theano.config.floatX)))
    params = initial_params
    op = model(X, params, hWidths, input_size, output_size)

    cost = T.sum(T.sqr(op - Y)) * (vy * 0.5) + T.sum(T.sqr(params ** 2)) * (precision * 0.5)
    updates = sgd(cost, params)
    # updates=Adam(cost,params)
    train = theano.function(inputs=[X, Y], outputs=cost,
                            updates=updates, allow_input_downcast=True,
                            name='train')
    predict = theano.function(inputs=[X], outputs=op, allow_input_downcast=True)
    fcost = theano.function(inputs=[op, Y], outputs=cost, allow_input_downcast=True)

    test_costs = []
    train_costs = []

    for i in range(epochs):
        for start, end in zip(range(0, len(X_train), mini_batchsize),
                              range(mini_batchsize, len(X_train), mini_batchsize)):
            yd = (floatX(y_train[start:end])).reshape(mini_batchsize, 1)
            cost_v = train(X_train[start:end], yd)

        # Done this cost prediction needs to change
        # fin_cost_test = fcost(predict(X_test), floatX(y_test).reshape(len(y_test), 1))
        # fin_cost_train = fcost(predict(X_train), floatX(y_train).reshape(len(y_train), 1))
        fin_cost_test = MSE(predict(X_test), y_test)
        fin_cost_train = MSE(predict(X_train), y_train)
        test_costs.append(fin_cost_test)
        train_costs.append(fin_cost_train)
        # print i, fin_cost_test, fin_cost_train

    # print 'final b_o values'
    # print b_o.get_value()

    # fin_cost_test = fcost(predict(X_test), floatX(y_test).reshape(len(y_test), 1))
    # fin_cost_train = fcost(predict(X_train), floatX(y_train).reshape(len(y_train), 1))
    fin_cost_test = MSE(predict(X_test), y_test)
    fin_cost_train = MSE(predict(X_train), y_train)
    print 'vy: {}, prec: {}, Train: {}, Test: {}'.format(vy, precision, fin_cost_train, fin_cost_test)

    # Calculate RMS error with simple mean prediction
    test_mean = np.mean(y_test)
    train_mean = np.mean(y_train)

    mean_p_test = np.ones(y_test.size) * test_mean
    mean_p_train = np.ones(y_train.size) * train_mean

    # test_cost=fcost(floatX(mean_p_test).reshape(len(y_test), 1), floatX(y_test).reshape(len(y_test), 1))
    # train_cost=fcost(floatX(mean_p_train).reshape(len(y_train), 1), floatX(y_train).reshape(len(y_train), 1))
    test_cost = MSE(mean_p_test, y_test)
    train_cost = MSE(mean_p_train, y_train)

    tArray = np.ones(epochs) * test_cost
    if (display):
        print 'MSE for mean prediction, Train:{} ,Test:{}'.format(train_cost, test_cost)

        plt.plot(range(epochs), test_costs, label='Train')
        plt.plot(range(epochs), train_costs, label='Train')

        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('TrainCost:{}, TestCost: {}'.format(fin_cost_train, fin_cost_test))

    return fin_cost_train, fin_cost_test


if __name__ == '__main__':
    mlp_synthetic()
