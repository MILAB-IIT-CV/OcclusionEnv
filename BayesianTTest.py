import os
import best
import matplotlib.pyplot as plt
import numpy as np
import ast

def makePlot(best_results, name, s, bins = 30):

    fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    #axes[0, 0].get_shared_x_axes().join(axes[0, 0], axes[1, 0])

    best.plot_posterior(best_results,
                   'Difference of means',
                   ax=axes,
                   bins=bins,
                   title='Difference of means',
                   stat='mean',
                   ref_val=0,
                   label=r'$\mu_1 - \mu_2$')

    fig.tight_layout()

    fig.savefig(name + "/" + s + "_DoM.pdf")

    fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    best.plot_posterior(best_results,
                   'Effect size',
                   ax=axes,
                   bins=bins,
                   title='Effect size',
                   ref_val=0,
                   label=r'$(\mu_1 - \mu_2) /'
                          r' \sqrt{(\mathrm{sd}_1^2 + \mathrm{sd}_2^2)/2}$')

    fig.tight_layout()

    fig.savefig(name + "/" + s + "_ES.pdf")

def getValsFromTest(best_results):

    hdi_min, hdi_max = best_results.hdi("Difference of means", 0.95)

    trace = best_results.trace['Difference of means']

    mean = trace.mean()
    median = np.median(trace)

    hist_vals, hist_bins = np.histogram(trace, bins=1000)
    pos_bins = (hist_bins >= 0)[:-1]
    ratio = np.sum(hist_vals[pos_bins]) / np.sum(hist_vals) * 100

    return hdi_min, hdi_max, ratio, mean, median

if __name__ == '__main__':

    base_path = "./results/test50"

    file = open(os.path.join(base_path + "/" + "iterations_raw_3.txt"), "r")
    iterations_raw = file.read()
    iterations_raw = np.array(list(ast.literal_eval(iterations_raw).values()))
    file.close()

    file = open(os.path.join(base_path + "/" + "iterations_random_3.txt"), "r")
    iterations_random = file.read()
    iterations_random = np.array(list(ast.literal_eval(iterations_random).values()))
    file.close()

    file = open(os.path.join(base_path + "/" + "iterations_model_3.txt"), "r")
    iterations_model = file.read()
    iterations_model = np.array(list(ast.literal_eval(iterations_model).values()))
    file.close()

    file = open(os.path.join(base_path + "/" + "iterations_model_3.txt"), "r")
    iterations_rl = file.read()
    iterations_rl = np.array(list(ast.literal_eval(iterations_rl).values()))
    file.close()


    print("Running BEST between Random and Raw")
    best_out = best.analyze_two(iterations_random, iterations_raw)
    hdi_min, hdi_max, ratio, mean, median = getValsFromTest(best_out)
    makePlot(best_out, base_path, "RandVSRaw")

    print("Running BEST between Random and Model")
    best_out = best.analyze_two(iterations_random, iterations_model)
    hdi_min, hdi_max, ratio, mean, median = getValsFromTest(best_out)
    makePlot(best_out, base_path, "RandVSModel")

    print("Running BEST between Random and RL")
    best_out = best.analyze_two(iterations_random, iterations_rl)
    hdi_min, hdi_max, ratio, mean, median = getValsFromTest(best_out)
    makePlot(best_out, base_path, "RandVSRL")

    print("Running BEST between Raw and Model")
    best_out = best.analyze_two(iterations_raw, iterations_model)
    hdi_min, hdi_max, ratio, mean, median = getValsFromTest(best_out)
    makePlot(best_out, base_path, "RawVSModel")

    print("Running BEST between Raw and RL")
    best_out = best.analyze_two(iterations_raw, iterations_rl)
    hdi_min, hdi_max, ratio, mean, median = getValsFromTest(best_out)
    makePlot(best_out, base_path, "RawVSRL")

    print("Running BEST between Model and RL")
    best_out = best.analyze_two(iterations_model, iterations_rl)
    hdi_min, hdi_max, ratio, mean, median = getValsFromTest(best_out)
    makePlot(best_out, base_path, "ModelVSRL")