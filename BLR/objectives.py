import numpy as np


# def objective(x):
#     return (-1) * np.cos(3*x)


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


"minimise (sin(10 *pi * x) / (2 * x)) + (x - 1)^4 x between 0.5 and 2.5 "

'minimise ((6x-2)^2)*sin(12x-4) x between 0 and 1 ' \
'' \
'' \
'minimise (-1)*((x-1)^2)*sin(3x+5/x+1) x between [5,10] '
