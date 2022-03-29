import re
from os import listdir
from os.path import isfile, join
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dpi = 300
nthreads = [1, 2, 4, 8, 16, 32, 48, 64]
datasets = [
    "covtype",
    "webspam",
    # "music",
    "rcv1",
    "epsilon",
    "news20"
]


def extract_epoch_time(f):
    with open(f, "r") as file:
        times = []
        epochs = []
        for line in file:
            l = line.strip()
            res = re.search(r'epoch_time: ([\d.]+)', l)
            if res:
                times.append(float(res.group(1)))
        times = times
        return np.array(times), np.array(epochs)


def extract_time(f):
    with open(f, "r") as file:
        times = []
        epochs = []
        for line in file:
            l = line.strip()
            res = re.search(r'epoch: (\d+) train_time: ([\d.]+)', l)
            if res:
                epochs.append(float(res.group(1)))
                times.append(float(res.group(2)))
        times = times
        epochs = epochs
        print(f, len(times))
        return np.array(times), np.array(epochs)


path = sys.argv[1]
numapath = ""  # sys.argv[2]
files = [f for f in listdir(path) if isfile(join(path, f))]
numafiles = []  # [f for f in listdir(numapath) if isfile(join(numapath, f))]
sns.set(style='whitegrid')


def plot_speedup(types, dataset, plot_name, index, log_y):
    fig, ax = plt.subplots()
    for name, threads in sorted(types.items()):
        xs = []
        ys = []
        for t, value in sorted(threads.items()):
            for y in value[index]:
                xs.append(t)
                ys.append(y)
        if log_y:
            sns.lineplot(x=np.log2(xs), y=np.log2(ys), label=name, markers=True, dashes=True, marker="o")
        else:
            sns.lineplot(x=np.log2(xs), y=ys, label=name, markers=True, dashes=True, marker="o")
        if name == "ideal" and len(ys) > 0:
            if log_y:
                plt.yticks(np.log2(ys), ys)
            else:
                plt.yticks(ys, ys)
    plt.xticks(np.log2(nthreads), nthreads)
    plt.legend()
    plt.title("%s Factor, %s" % (plot_name, dataset))
    plt.xlabel("Threads")
    plt.ylabel("Factor")
    fig.tight_layout()
    fig.savefig('%s_%s_factor.png' % (dataset, plot_name), dpi=dpi)


for dataset in datasets:
    types = {"hogwild": {}}

    for f in files:
        res = re.search(r'%s_(\d+)_[\d.]+_[\d.]+.txt' % dataset, f)
        if res:
            threads = int(res.group(1))
            times, epochs = extract_time(join(path, f))
            iteration_times, _ = extract_epoch_time(join(path, f))
            if len(times) == 0:
                continue
            types["hogwild"][threads] = times, epochs, iteration_times

    for f in numafiles:
        res = re.search(r'%s_(\d+)_(\d+)_[\d.]+_[\d.]+.txt' % dataset, f)
        if res:
            threads = int(res.group(1))
            c = int(res.group(2))
            name = "hogwild++%2d" % c
            if name not in types:
                types[name] = {}
            times, epochs = extract_time(join(numapath, f))
            iteration_times, _ = extract_epoch_time(join(numapath, f))
            if len(times) < 90:
                continue
            types[name][threads] = times, epochs, iteration_times

    basic_time = np.mean(types["hogwild"][1][0])
    basic_epoch = np.mean(types["hogwild"][1][1])
    basic_iteration_time = np.mean(types["hogwild"][1][2])

    for name, threads in types.items():
        for t, results in threads.items():
            times, epochs, iteration_times = results
            threads[t] = basic_time / times, epochs / basic_epoch, basic_iteration_time / iteration_times

    types["ideal"] = {}
    for t in nthreads:
        types["ideal"][t] = [t], [], [t]

    plot_speedup(types, dataset, "Convergence", 0, True)
    plot_speedup(types, dataset, "Epochs", 1, False)
    plot_speedup(types, dataset, "Iteration time", 2, True)
