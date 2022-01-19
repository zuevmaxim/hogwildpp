import re
from os import listdir
from os.path import isfile, join
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dpi = 300

datasets = [
    # "covtype",
    # "webspam",
    # "music",
    "rcv1",
    # "epsilon",
    # "news20"
]


def extract_epoch_time(f):
    with open(f, "r") as file:
        times = []
        epochs = []
        for line in file:
            l = line.strip()
            res = re.search('epoch_time: ([0-9\.]+)', l)
            if res:
                times.append(float(res.group(1)))
        times = times
        print(f, len(times))
        return times, epochs

def extract_time(f):
    with open(f, "r") as file:
        times = []
        epochs = []
        for line in file:
            l = line.strip()
            res = re.search('epoch: ([0-9]+) train_time: ([0-9\.]+)', l)
            if res:
                epochs.append(float(res.group(1)))
                times.append(float(res.group(2)))
        times = times
        epochs = epochs
        print(f, len(times))
        return times, epochs


path = sys.argv[1]
numapath = sys.argv[2]
files = [f for f in listdir(path) if isfile(join(path, f))]
numafiles = [f for f in listdir(numapath) if isfile(join(numapath, f))]
sns.set(style='whitegrid')


def plot_time(types, dataset, plot_name):
    fig, ax = plt.subplots()
    for name, threads in sorted(types.items()):
        xs = []
        ys = []
        for t, results in threads.items():
            time_ms = results[0]
            for tms in time_ms:
                xs.append(t)
                ys.append(tms)
        sns.lineplot(np.log2(xs), ys, label=name, marker="o", ci=95)
        if name == "hogwild":
            plt.xticks(np.log2(xs), xs)
    plt.title("%s Time, %s" % (plot_name, dataset))
    plt.xlabel("Threads")
    plt.ylabel("Time, s")
    fig.tight_layout()
    fig.savefig('%s_%s_time.png' % (dataset, plot_name), dpi=dpi)


def plot_iterations(types, dataset, plot_name):
    fig, ax = plt.subplots()
    min_y, max_y = int(1e9), 0
    for name, threads in sorted(types.items()):
        xs = []
        ys = []
        for t, results in threads.items():
            iterations = results[1]
            for i in iterations:
                xs.append(t)
                ys.append(i)
        min_y = min(min_y, min(ys))
        max_y = max(max_y, max(ys))
        sns.lineplot(np.log2(xs), np.log2(ys), label=name, marker="o", ci=95)
        if name == "hogwild":
            plt.xticks(np.log2(xs), xs)
    tiks = [2 ** i for i in range(int(np.floor(np.log2(min_y))), int(np.ceil(np.log2(max_y))))]
    plt.yticks(np.log2(tiks), tiks)
    plt.title("Iterations to converge, %s" % dataset)
    plt.xlabel("Threads")
    plt.ylabel("Iterations")
    fig.tight_layout()
    fig.savefig('%s_%s_iterations.png' % (dataset, plot_name), dpi=dpi)


def plot_speedup(types, dataset, plot_name):
    fig, ax = plt.subplots()
    for name, threads in sorted(types.items()):
        xs = []
        ys = []
        for t, time_ms in threads.items():
            xs.append(t)
            ys.append(base_ms / time_ms)

        xs, ys = zip(*sorted(zip(xs, ys)))
        ax.plot(np.log2(xs), np.log2(ys), 'o-', label=name)
        if name == "ideal":
            plt.xticks(np.log2(xs), xs)
            plt.yticks(np.log2(ys), ys)
    plt.legend()
    plt.title("%s Speed-up, %s" % (plot_name, dataset))
    plt.xlabel("Threads")
    plt.ylabel("Speed-up")
    fig.tight_layout()
    fig.savefig('%s_%s_speed-up.png' % (dataset, plot_name), dpi=dpi)


for dataset in datasets:
    types = {}

    types["hogwild"] = {}
    for f in files:
        res = re.search('%s_([0-9]+)_[0-9\.]+_[0-9\.]+.txt' % dataset, f)
        if res:
            threads = int(res.group(1))
            types["hogwild"][threads] = extract_time(join(path, f))

    for f in numafiles:
        res = re.search('%s_([0-9]+)_([0-9]+)_[0-9\.]+_[0-9\.]+.txt' % dataset, f)
        if res:
            threads = int(res.group(1))
            c = int(res.group(2))
            name = "hogwild++%2d" % c
            if name not in types:
                types[name] = {}
            types[name][threads] = extract_time(join(numapath, f))

    plot_time(types, dataset, "Convergence")
    plot_iterations(types, dataset, "Convergence")

    for name, threads in types.items():
        for t, results in threads.items():
            time_ms, epochs = results
            threads[t] = np.average(time_ms)
    base_ms = types["hogwild"][1]
    types["ideal"] = {}
    for t in types["hogwild"].keys():
        types["ideal"][t] = base_ms / t

    plot_speedup(types, dataset, "Convergence")

for dataset in datasets:
    types = {}

    types["hogwild"] = {}
    for f in files:
        res = re.search('%s_([0-9]+)_[0-9\.]+_[0-9\.]+.txt' % dataset, f)
        if res:
            threads = int(res.group(1))
            types["hogwild"][threads] = extract_epoch_time(join(path, f))

    for f in numafiles:
        res = re.search('%s_([0-9]+)_([0-9]+)_[0-9\.]+_[0-9\.]+.txt' % dataset, f)
        if res:
            threads = int(res.group(1))
            c = int(res.group(2))
            name = "hogwild++%2d" % c
            if name not in types:
                types[name] = {}
            types[name][threads] = extract_epoch_time(join(numapath, f))

    plot_time(types, dataset, "Iteration")

    for name, threads in types.items():
        for t, results in threads.items():
            time_ms, epochs = results
            threads[t] = np.average(time_ms)
    base_ms = types["hogwild"][1]
    types["ideal"] = {}
    for t in types["hogwild"].keys():
        types["ideal"][t] = base_ms / t

    plot_speedup(types, dataset, "Iteration")
