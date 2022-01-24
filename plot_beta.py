import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set(style='whitegrid')


def find_root(f, start, end, eps):
    err = 1e9
    m = 0
    while abs(err) > eps or err > 0:
        m = (start + end) / 2
        err = f(m)
        if err > 0.0:
            end = m
        else:
            start = m
    return m


def iterations(beta, m, n):
    s = beta
    result = [s]
    for i in range(1, n):
        s += beta ** (i * m)
        result.append(s)
    return result


n = 10

colors = ['b', 'g', 'r', 'm']
for i, m in enumerate([2, 4, 16, 64]):
    beta = find_root(lambda x: x + x ** m - 1, 0, 1, 0.00001)
    l = 1 - beta ** (m - 1)
    print(m, beta, l)
    # ys = iterations(beta, m, n)
    # sns.lineplot(x=list(range(len(ys))), y=ys, label=m, color=colors[i], marker="o")
    #
    # ys = np.array(ys) / (beta + (beta ** m) / (1 - beta ** m))
    # sns.lineplot(x=list(range(len(ys))), y=ys, color=colors[i], marker="o", linestyle='--')
    #
    # beta = find_root(lambda x: x + x ** m / (1 - x ** m) - 1, 0, 1, 0.00001)
    # ys = iterations(beta, m, n)
    # sns.lineplot(x=list(range(len(ys))), y=ys, color=colors[i], linestyle='-.', marker="o")

# plt.ylabel("One update contribution")
# plt.xlabel("Number of synchronization rounds")
# plt.savefig("beta.png", dpi=300)

