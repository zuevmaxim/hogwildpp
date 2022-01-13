#!/usr/bin/env python2

import os, sys, math, time, subprocess, multiprocessing
from subprocess import check_call

dryrun = False

dataset = [
	# "covtype",
	"webspam",
	# "music",
	# "rcv1",
	"epsilon",
	"news20"
]
# settings used for grid size search
'''
nthreads = [10]
maxstepsize = {	"covtype" : 5e-2,
		"webspam" : 5e+0,
		"music"   : 5e-8,
		"rcv1"    : 1e-0,
		"epsilon" : 1e-1,
		"news20"  : 1e-0,
	   }
iterations = {	"default" : 200, "news20"  : 350, "epsilon" : 150}
stepdecay = [1, 0.95, 0.9, 0.85, 0.8]
stepdecay_per_dataset = {}
step_search_range = 10
'''
# settings used for collecting results
nthreads = [1, 2, 4, 8, 16, 32, 48, 64]
maxstepsize = { "covtype" : 5e-03,
		"webspam" : 2e-01,
		"music"   : 5e-08,
		"rcv1"    : 5e-01,
		"epsilon" : 1e-01,
		"news20"  : 5e-01,
	      }
target_accuracy = { "covtype" : 1,
                    "webspam" : 0.92854,
                    "rcv1"    : 0.97713,
                    "epsilon" : 0.8974,
                    "news20"  : 0.99925,
                    }
stepdecay = []
stepdecay_per_dataset = { "covtype" : [0.85],
			  "webspam" : [0.8],
			  "music"   : [0.8],
			  "rcv1"    : [0.8],
			  "epsilon" : [0.85],
			  "news20"  : [0.8],
			}
iterations = {	"default" : 150, "epsilon" : 75}
step_search_range = 0

outputdir = "svm_" + time.strftime("%m%d-%H%M%S")

if len(sys.argv) > 1:
	if sys.argv[1] == "-n":
		dryrun = True
	if sys.argv[1] == "-y":
		dryrun = False

if not dryrun:
	check_call("mkdir -p {}/".format(outputdir), shell=True)

def GenerateSteps(max_step_size):
	return [max_step_size]

for d in dataset:
	# Find a step size from table
	steps = GenerateSteps(maxstepsize[d])
	if d in iterations:
		epochs = iterations[d]
	else:
		epochs = iterations["default"]
	print "For dataset {} we will use {} epochs and step size:\n {}\n".format(d, epochs, steps)
	for s in steps:
		for n in nthreads:
			if d in stepdecay_per_dataset:
				stepdecay_trials = stepdecay_per_dataset[d]
			else:
				stepdecay_trials = stepdecay
			for b in stepdecay_trials:
				result_name = os.path.join(outputdir, "{}_{}_{}_{}.txt".format(d, n, s, b))
				cmdline = "bin/svm --epoch {} --stepinitial {} --step_decay {} --split {} --target_accuracy {} data/{}_train.tsv data/{}_test.tsv | tee {}".format(epochs, s, b, n, target_accuracy[dataset], d, d, result_name)
				print "Executing HogWild! with {} threads:\n{}\nResults at {}".format(n, cmdline, result_name)
				if not dryrun:
						subprocess.Popen(cmdline, shell=True).wait()
				else:
					print "*** This is a dry run. No results will be produced. ***"
	print


