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
    x[0,:] accesses all the coordinates of the second data-pt
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

"minimise (sin(10 *pi * x) / (2 * x)) + (x - 1)^4 x between 0.5 and 2.5 "

'minimise ((6x-2)^2)*sin(12x-4) x between 0 and 1 ' \
'' \
'' \
'minimise (-1)*((x-1)^2)*sin(3x+5/x+1) x between [5,10] '

if __name__ == '__main__':
    print brannin_hoo(np.asarray([[-pi, 12.275],
                                  [9.42478, 2.475],
                                  [3, 3]]))

    print np.asarray([[-pi, 12.275],
                      [9.42478, 2.475],
                      [3, 3]]).shape
