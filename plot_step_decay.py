import math
import re
import sys
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from common import datasets, extract_time, extract_epoch_time, get_step_decay, dpi

sns.set(style='whitegrid')

basic_path = sys.argv[1]
path = sys.argv[2]
numapath = sys.argv[3]
# updatedpath = sys.argv[4]
files = [f for f in listdir(path) if isfile(join(path, f))]
numafiles = [f for f in listdir(numapath) if isfile(join(numapath, f))]
# updatedfiles = [f for f in listdir(updatedpath) if isfile(join(updatedpath, f))]

for dataset in datasets:
    results = []
    row_results = []
    basic_step_decay = get_step_decay(dataset)

    basic_file = [f for f in listdir(basic_path) if isfile(join(basic_path, f))][0]
    times, epochs = extract_time(join(basic_path, basic_file))
    iteration_times, _ = extract_epoch_time(join(basic_path, basic_file))
    basic_time = np.average(times)
    basic_epochs = np.average(epochs)
    basic_iteration_time = np.average(iteration_times)

    for f in files:
        res = re.search(r'%s_\d+_([\d.]+).txt' % dataset, f)
        if res:
            step_decay = float(res.group(1))
            times, epochs = extract_time(join(path, f))
            iteration_times, _ = extract_epoch_time(join(path, f))
            if len(times) == 0:
                continue
            step_decay_factor = math.log(step_decay, basic_step_decay)
            res = "HogWild", step_decay_factor, basic_time / np.average(times), np.average(epochs) / basic_epochs, \
                  basic_iteration_time / np.average(iteration_times), len(times)
            results.append(res)
            for t, e in zip(times, epochs):
                row_results.append(("HogWild  ", step_decay_factor, basic_time / t, e / basic_epochs))

    for f in numafiles:
        res = re.search(r'%s_\d+_\d+_([\d.]+)_([\d.]+).txt' % dataset, f)
        if res:
            step_decay = float(res.group(1))
            sync_delay = int(float(res.group(2)))
            if sync_delay not in [512, 8, 64]:
                continue
            times, epochs = extract_time(join(numapath, f))
            iteration_times, _ = extract_epoch_time(join(numapath, f))
            if len(times) == 0:
                continue
            step_decay_factor = math.log(step_decay, basic_step_decay)
            res = "HogWild++, {}".format(sync_delay), step_decay_factor, basic_time / np.average(times), np.average(epochs) / basic_epochs, \
                  basic_iteration_time / np.average(iteration_times), len(times)
            results.append(res)
            for t, e in zip(times, epochs):
                row_results.append(("HogWild++, {}".format(sync_delay), step_decay_factor, basic_time / t, e / basic_epochs))
    results.sort()
    df = pd.DataFrame(results, columns=['algorithm', 'step_decay_power', 'time_speed-up', 'iterations_factor', 'iteration_time_speed-up', 'tests'])
    for param in ['iteration_time_speed-up', 'tests']:
        fig, ax = plt.subplots()
        data = df[df['algorithm'] != "HogWild  "] if param == 'iteration_time_speed-up' else df
        sns.lineplot(data=data, x='step_decay_power', y=param, hue='algorithm', markers=True, dashes=True, marker="o")
        plt.xlabel('Step decay power')
        plt.ylabel(param)
        plt.title("{} {}, 2 NUMA with 64x2 threads".format(dataset, param))
        fig.savefig('pictures/{}_{}.png'.format(dataset, param), dpi=dpi)

    df = pd.DataFrame(row_results, columns=['algorithm', 'step_decay_power', 'time_speed-up', 'iterations_factor'])
    for param in ['time_speed-up', 'iterations_factor']:
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x='step_decay_power', y=param, hue='algorithm', markers=True, dashes=True, marker="o")
        plt.xlabel('Step decay power')
        plt.ylabel(param)
        plt.title("{} {}, 2 NUMA with 64x2 threads".format(dataset, param))
        fig.savefig('pictures/{}_{}.png'.format(dataset, param), dpi=dpi)



    # xs = [x[1] for x in results]
    # ys = [1 / y[2] for y in results]
    # fig, ax = plt.subplots()
    # sns.lineplot(x=xs, y=ys, markers=True, dashes=True, marker="o")
    # fig.savefig('plot.png', dpi=dpi)

    for result in results:
        name, step_decay_factor, time, epoch, iteration_time, length = result
        print("{} Factor: {:.1f} Speed-up: {:6.3f} Iterations factor: {:6.3f} Iteration time speed-up: {:.3f} Tests: {}"
              .format(name, step_decay_factor, time, epoch, iteration_time, length))
