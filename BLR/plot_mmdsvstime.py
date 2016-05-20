__author__ = 'nt357'
from matplotlib.pyplot import figure, show, legend, xlabel, ylabel, errorbar, xlim, ylim, savefig, fill, fill_between
from pickle import load
import matplotlib as mpl
import numpy as np
import os
from collections import OrderedDict


def plot_mmd_fig(data_set=0, feature_num=2, show_plot=False, save_fig=True):

    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.unicode'] = True

    #sampler_names = ["HMC", "binary_b", "standard"]

    colours = ['blue', 'red', 'magenta', 'green']

    folder = "Experiments"
    data_names = ["MMDforMof2GBinary_B_eps_1.9_L_40.pkl", "MMDforMof2GBinary_B_eps_0.5_L_100.pkl",
                  "MMDforMof2GBinary_B_eps_1.0_L_50.pkl", "MMDforMof2GBinary_B_eps_1.5_L_33.pkl"
                  ]
    data_paths = [os.path.join(folder, i) for i in data_names]

    f = open(data_paths[data_set], "r")
    all_data = load(f)
    metrics = ["mean_errors", "cov_errors", "mmd_errors"]

    list_of_Bs = []
    ii = 0

    for key in sorted(all_data.keys()):
        c = float(key.split()[-1])
        name = "B = "
        list_of_Bs.append(name+str(c))
        data = all_data[key]
        # np.mean(data["acc_ratio"])
        trials, time = data[metrics[feature_num]].shape
        times = 1000*np.array(range(1, 16))

        stats = data[metrics[feature_num]]
        aver = np.mean(stats, axis=0)
        stds = .5*np.std(stats, axis=0)


        figure(1)
        errorbar(times, aver, yerr=stds, fmt="o-", color=colours[ii])
        ii += 1

    legend(list_of_Bs)
    xlabel("Number of Samples", fontsize=16)

    xlim(500, 15500)
    ylim(-0.25, 30)
    mpl.pyplot.axes().set_aspect('equal')

    if metrics[feature_num] == "mean_errors":
        ylabel("absolute mean error", fontsize=16)
    if metrics[feature_num] == "cov_errors":
        ylabel("absolute covariance error", fontsize=16)
    if metrics[feature_num] == "mmd_errors":
        ylabel("Maximum Mean Discrepancy", fontsize=16)
    #

    if save_fig:
        save_folder = "Figures/Mof2G"
        fig_name = data_names[data_set][:-4]+".pdf"
        save_path = os.path.join(save_folder, fig_name)
        savefig(save_path, bbox_inches='tight')

    if show_plot:
        show()

if __name__ == "__main__":
    plot_mmd_fig(data_set=3, feature_num=2, show_plot=False, save_fig=True)