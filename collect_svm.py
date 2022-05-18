#!/usr/bin/env python3

import os
import subprocess
import time
from subprocess import check_call

from common import datasets, is_dry_run, get_epochs, maxstepsize, create_step_decay_trials, target_accuracy

nthreads = [128]
iterations = {"default": 150, "epsilon": 75}
outputdir = "results/svm_" + time.strftime("%m%d-%H%M%S")

if not is_dry_run():
    check_call("mkdir -p {}/".format(outputdir), shell=True)

for d in datasets:
    s = maxstepsize[d]
    epochs = get_epochs(d, iterations)
    for n in nthreads:
        for b in create_step_decay_trials(d, n):
            result_name = os.path.join(outputdir, "{}_{}_{}.txt".format(d, n, b))
            cmdline = "bin/svm --epoch {} --stepinitial {} --step_decay {} --split {} --target_accuracy {} data/{}_train.tsv data/{}_test.tsv | tee {}".format(
                epochs, s, b, n, target_accuracy[d], d, d, result_name)
            print("Executing HogWild! with {} threads:\n{}\nResults at {}".format(n, cmdline, result_name))
            if not is_dry_run():
                subprocess.Popen(cmdline, shell=True).wait()
            else:
                print("*** This is a dry run. No results will be produced. ***")
    print()
