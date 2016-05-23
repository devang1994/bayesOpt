from bayesOptBNN import bayes_opt
import matplotlib.pyplot as plt
import objectives

func = objectives.objectiveSinCos
xr = [-1, 1]
actual_min = -6.02074
numDim = 1
init_random = 20
k = 10
# num_it=18
num_it = 1
numDim = len(xr) / 2
seed=1020


bVals = bayes_opt(func, xr, initial_random=init_random, num_it=num_it, k=k, hWidths=[50, 50, 50],
                  precisions=[1, 1, 1, 1], vy=100,
                  show_evo=True, actual_min=actual_min, numDim=numDim, seed=seed)

plt.show()