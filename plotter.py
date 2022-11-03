import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == '__main__':
    trainIdx = 26

    folder = "./PPO_logs/OcclusionEnv/"
    file = folder + "PPO_OcclusionEnv_log_" + str(trainIdx) + ".csv"

    otherfile_raw = "./results/test2022-11-03 08:38:11.668158/iterations_raw_3.txt"
    otherfile_random = "results/test2022-11-03 08:38:11.668158/iterations_random_3.txt"
    otherfile_model = "results/test2022-11-03 08:38:11.668158/iterations_model_3.txt"

    raw_str = open(otherfile_raw).read()
    raw = np.array(list(ast.literal_eval(raw_str).values()))
    raw_mean = np.mean(raw)
    raw_std = np.std(raw)
    random_str = open(otherfile_random).read()
    random = np.array(list(ast.literal_eval(random_str).values()))
    random_mean = np.mean(random)
    random_std = np.std(random)
    model_str = open(otherfile_model).read()
    model = np.array(list(ast.literal_eval(model_str).values()))
    model_mean = np.mean(model)
    model_std = np.std(model)

    data = np.genfromtxt(file, dtype=None, delimiter=',', names=True)

    running_rew = data['running_reward']
    times = data['length']

    ma_times = moving_average(times, 50)

    ts = pd.Series(times)
    ma_times2 = ts.rolling(50).mean()
    ms_times2 = ts.rolling(50).std()

    print(raw_mean, random_mean, model_mean, np.min(ma_times2))
    print(raw_std, random_std, model_std, np.min(ms_times2))

    plt.figure()
    plt.title("Running Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.plot(running_rew)
    plt.figure()
    plt.title("Timesteps")
    plt.xlabel("Episodes")
    plt.ylabel("Timesteps")
    plt.plot(ma_times)
    plt.show()