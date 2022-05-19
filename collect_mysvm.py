#!/usr/bin/env python3

import os
import subprocess
import time
from subprocess import check_call

from common import datasets, maxstepsize, target_accuracy, get_epochs, is_dry_run, \
    create_step_decay_trials


def generate_update_delays(nweights):
    if nweights <= 4:
        update_delay = 64
    elif nweights <= 10:
        update_delay = 16
    else:
        update_delay = 4
    return [update_delay * (2 ** i) for i in range(-3, 4)]


nthreads = [128]
cluster_size = [32]
iterations = {"default": 50, "epsilon": 25}
outputdir = "results/mysvm_" + time.strftime("%m%d-%H%M%S")

if not is_dry_run():
    check_call("mkdir -p {}/".format(outputdir), shell=True)

for d in datasets:
    s = maxstepsize[d]
    epochs = get_epochs(d, iterations)
    for n in nthreads[::-1]:
        for c in cluster_size[::-1]:
            nweights = n / c
            if (n % c) != 0:
                continue
            effective_epochs = epochs * nweights
            effective_epochs = min(1000, effective_epochs)
            effective_epochs = max(150, effective_epochs)

            for u in generate_update_delays(nweights):
                for b in create_step_decay_trials(d, n):
                    result_name = os.path.join(outputdir, "{}_{}_{}_{}_{}.txt".format(d, n, c, b, u))
                    cmdline = "bin/mysvm --epoch {} --stepinitial {} --step_decay {} --update_delay {} --cluster_size {} --split {}  --target_accuracy {} data/{}_train.tsv data/{}_test.tsv | tee {}".format(
                        effective_epochs, s, b, u, c, n, target_accuracy[d], d, d, result_name)
                    print("Executing HogWild++ with {} threads, c={}:\n{}\nResults at {}".format(n, c, cmdline,
                                                                                                 result_name))
                    if not is_dry_run():
                        subprocess.Popen(cmdline, shell=True).wait()
                    else:
                        print("*** This is a dry run. No results will be produced. ***")
    print()
