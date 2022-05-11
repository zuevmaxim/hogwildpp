#!/usr/bin/env python2

import os, sys, math, time, subprocess, multiprocessing
from subprocess import check_call

dryrun = False

datasets = [
	# "covtype",
	# "webspam",
	# "music",
	"rcv1",
	# "epsilon",
	# "news20"
]
# settings used for grid size search
'''
nthreads = [10]
iterations = {	"default" : 200, "news20"  : 350, "epsilon" : 150}
maxstepsize = { "covtype" : 5e-03,
		"webspam" : 2e-01,
		"music"   : 5e-08,
		"rcv1"    : 5e-01,
		"epsilon" : 1e-01,
		"news20"  : 5e-01,
	      }
stepdecay = [1, 0.95, 0.9, 0.85, 0.8]
stepdecay_per_dataset = {}
step_search_range = 10
'''
nthreads = [16]
cluster_size = [2]
maxstepsize = { "covtype" : 5e-03,
		"webspam" : 2e-01,
		"music"   : 5e-08,
		"rcv1"    : 5e-01,
		"epsilon" : 1e-01,
		"news20"  : 5e-01,
	      }
target_accuracy = { "covtype" : 0.76291,
					"webspam" : 0.92700,
					"rcv1"    : 0.97713,
					"epsilon" : 0.89740,
					"news20"  : 0.96425,
					}
stepdecay = []
stepdecay_per_dataset = { "covtype" : [0.85],
			  "webspam" : [0.8],
			  "music"   : [0.8],
			  "rcv1"    : [0.8],
			  "epsilon" : [0.85],
			  "news20"  : [0.8],
			}
iterations = {	"default" : 50, "epsilon" : 25}
step_search_range = 0

outputdir = "mysvm_" + time.strftime("%m%d-%H%M%S")

if len(sys.argv) > 1:
	if sys.argv[1] == "-n":
		dryrun = True
	if sys.argv[1] == "-y":
		dryrun = False

if not dryrun:
	check_call("mkdir -p {}/".format(outputdir), shell=True)

def GenerateSteps(max_step_size):
	return [max_step_size]

def GenerateUpdateDelay(nweights):
	if nweights <= 4:
		update_delay = 64
	elif nweights <= 10:
		update_delay = 16
	else:
		update_delay = 4
	return update_delay

for d in datasets:
	# Find a step size from table
	steps = GenerateSteps(maxstepsize[d])
	if d in iterations:
		epochs = iterations[d]
	else:
		epochs = iterations["default"]
	print("For dataset {} we will use {} epochs and step size:\n {}\n".format(d, epochs, steps))
	for s in steps:
		for n in nthreads[::-1]:
			for c in cluster_size[::-1]:
				nweights = n / c
				if (n % c) != 0:
					continue
				effective_epochs = epochs * nweights
				effective_epochs = min(1000, effective_epochs)
				effective_epochs = max(150, effective_epochs)
				u = GenerateUpdateDelay(nweights) * 8
				if d in stepdecay_per_dataset:
					stepdecay_trials = stepdecay_per_dataset[d]
				else:
					stepdecay_trials = stepdecay
				for b in stepdecay_trials:
					effective_b = math.pow(b, (1.0 / nweights))
					result_name = os.path.join(outputdir, "{}_{}_{}_{}_{}.txt".format(d, n, c, s, b))
					cmdline = "bin/mysvm --epoch {} --stepinitial {} --step_decay {} --update_delay {} --cluster_size {} --split {}  --target_accuracy {} data/{}_train.tsv data/{}_test.tsv | tee {}".format(effective_epochs, s, effective_b, u, c, n, target_accuracy[d], d, d, result_name)
					print("Executing HogWild++ with {} threads, c={}:\n{}\nResults at {}".format(n, c, cmdline, result_name))
					if not dryrun:
						subprocess.Popen(cmdline, shell=True).wait()
					else:
						print("*** This is a dry run. No results will be produced. ***")
	print()



