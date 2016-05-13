import numpy as np


# def objective(x):
#     return (-1) * np.cos(3*x)


def objectiveSinCos(x):
    return (np.sin(x * 7) + np.cos(x * 17))


def objectiveCos(x):
    return (-1) * np.cos(3 * x)


def objectiveGramacyLee(x):
    return (np.sin(10 * np.pi * x) / (2 * x)) + np.power((x - 1), 4)
