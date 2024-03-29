import numpy as np


# def objective(x):
#     return (-1) * np.cos(3*x)

pi = np.pi

pi_sqr = np.square(pi)

def objectiveSinCos(x):
    return (np.sin(x * 7) + np.cos(x * 17))


def objectiveCos(x):
    return (-1) * np.cos(3 * x)


def objectiveGramacyLee(x):
    return (np.sin(10 * np.pi * x) / (2 * x)) + np.power((x - 1), 4)


def objectiveForrester(x):
    return (np.square(6 * x - 2)) * (np.sin(12 * x - 4))


def syntheticSinusoidal(x):
    return (np.square(x - 1)) * (np.sin(3 * x + 1 + 5 / x)) * (-1)


def brannin_hoo(x, a=1.0, b=5.1 / (4.0 * pi_sqr), c=5.0 / (pi), r=6.0, s=10.0, t=1.0 / (8.0 * pi)):
    """
    format of array([[1, 1],
       [2, 2],
       [3, 3]])

    [1,1] is first data pt etc
    x[:,0] acceses the first co-ordinate of each pt
    x[0,:] accesses all the coordinates of the first data-pt
    shape(x) (numTP,numD), numD =2


    :param x:
    :return:
    """
    # print 'in obj brannin'
    # print x
    # print x.shape
    ntrain = x.shape[0]
    x1 = x[:, 0]
    x2 = x[:, 1]
    t1 = x2 - b * np.square(x1) - r + c * x1
    t1 = np.square(t1)
    t1 = a * t1
    t2 = s + s * (1 - t) * np.cos(x1)
    t = (t1 + t2).reshape(ntrain, 1)
    return t


def rosenbrock_2D(x, a=1, b=100):
    ntrain = x.shape[0]
    x1 = x[:, 0]
    x2 = x[:, 1]

    t = np.square(a - x1) + b * np.square(x2 - np.square(x1))
    t = t.reshape(ntrain, 1)
    return t


def rosenbrock_2D_noisy(x, a=1, b=100, sd=0.1):
    ntrain = x.shape[0]
    x1 = x[:, 0]
    x2 = x[:, 1]

    t = np.square(a - x1) + b * np.square(x2 - np.square(x1))
    t = t.reshape(ntrain, 1)
    noise = np.random.normal(0, sd, ntrain).reshape(ntrain, 1)
    t = t + noise
    return t


def modified_rescaled_brannin_hoo(x, a=(1.0) / (51.95), b=5.1 / (4.0 * pi_sqr), c=5.0 / (pi), r=6.0, s=10.0,
                                  t=1.0 / (8.0 * pi), s1=44.81):
    """
    format of array([[1, 1],
       [2, 2],
       [3, 3]])

    [1,1] is first data pt etc
    x[:,0] acceses the first co-ordinate of each pt
    x[0,:] accesses all the coordinates of the first data-pt
    shape(x) (numTP,numD), numD =2


    :param x:
    :return:
    """
    # print 'in obj brannin'
    # print x
    # print x.shape
    ntrain = x.shape[0]
    x1 = x[:, 0]
    x2 = x[:, 1]
    t1 = x2 - b * np.square(x1) - r + c * x1
    t1 = np.square(t1)
    t1 = a * t1
    t2 = a * (s * (1 - t) * np.cos(x1) - s1)
    t = (t1 + t2).reshape(ntrain, 1)
    return t


def sixhumpcamel(x):
    """
    format of array([[1, 1],
       [2, 2],
       [3, 3]])

    [1,1] is first data pt etc
    x[:,0] acceses the first co-ordinate of each pt
    x[0,:] accesses all the coordinates of the first data-pt
    shape(x) (numTP,numD), numD =2


    :param x:
    :return:
    """
    # print 'in obj brannin'
    # print x
    # print x.shape
    ntrain = x.shape[0]
    x1 = x[:, 0]
    x2 = x[:, 1]
    t1 = 4.0 - 2.1 * np.square(x1) + np.square(np.square(x1)) / 3.0
    t1 = t1 * np.square(x1)
    t1 = t1 + x1 * x2
    t2 = 4 * x2 * x2 - 4
    t2 = t2 * x2 * x2
    t = (t1 + t2).reshape(ntrain, 1)
    return t


def mccormick(x):
    """
    format of array([[1, 1],
       [2, 2],
       [3, 3]])

    [1,1] is first data pt etc
    x[:,0] acceses the first co-ordinate of each pt
    x[0,:] accesses all the coordinates of the first data-pt
    shape(x) (numTP,numD), numD =2


    :param x:
    :return:
    """
    # print 'in obj brannin'
    # print x
    # print x.shape
    ntrain = x.shape[0]
    x1 = x[:, 0]
    x2 = x[:, 1]
    t1 = np.sin(x1 + x2)
    t2 = (x1 - x2) ** 2
    t2 = t2 + 2.5 * x2 - 1.5 * x1 + 1
    t = (t1 + t2).reshape(ntrain, 1)
    return t

def nonSmooth1d(x):

    # return (np.sin(x * 7) + np.cos(x * 17))
    out=(np.ones(x.shape[0])).reshape(-1,1)
    for i in range(x.shape[0]):
        if(0<=x[i]<0.25):
            out[i]=1
        elif(0.25<=x[i]<0.4):
            out[i] = -0.5
        elif(0.4<=x[i]<0.5):
            out[i] = -1
        elif (0.5 <= x[i] <= 1):
            out[i] = 1
        else:
            out[i]=-100

    return out


"minimise (sin(10 *pi * x) / (2 * x)) + (x - 1)^4 x between 0.5 and 2.5 "

'minimise ((6x-2)^2)*sin(12x-4) x between 0 and 1 ' \
'' \
'' \
'minimise (-1)*((x-1)^2)*sin(3x+5/x+1) x between [5,10] '
import matplotlib.pyplot as plt
if __name__ == '__main__':
    # print brannin_hoo(np.asarray([[-pi, 12.275],
    #                               [9.42478, 2.475],
    #                               [3, 3]]))

    # print np.asarray([[-pi, 12.275],
    #                   [9.42478, 2.475],
    #                   [3, 3]]).shape

    a = (np.arange(0, 1, 0.01).reshape(-1,1))
    print a.shape
    b = nonSmooth1d(a)

    print b.shape
    print b

    print np.hstack((a,b))
    plt.plot(a[:],b[:])
    plt.show()
    pass
