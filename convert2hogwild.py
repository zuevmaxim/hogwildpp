#!/usr/bin/env pypy

import os
import random
import sys


def convert_to_tsv(lines, out):
    index = 0
    for line in lines:
        elements = line.split()
        print("{}\t{}\t{}".format(index, -2, elements[0]), file=out)
        for values in elements[1:]:
            pairs = values.split(':')
            if len(pairs) == 1:
                print("Error parsing value pair: {}".format(values))
                continue
            print("{}\t{}\t{}".format(index, int(pairs[0]) - 1, pairs[1]), file=out)
        index += 1
        if index & 0xfff == 0xfff:
            sys.stdout.write('.')
            sys.stdout.flush()


if len(sys.argv) < 2:
    print("Usage: {} <input binary> <output tsv>".format(sys.argv[0]))
    os._exit(0)

input_file = sys.argv[1]
output_file = sys.argv[2]
split = len(sys.argv) > 3 and sys.argv[3] == "--split"

with open(input_file, 'r') as f:
    if split:
        lines = f.readlines()
        random.shuffle(lines)
        train_out = open(output_file + "_train.tsv", 'w')
        test_out = open(output_file + "_test.tsv", 'w')
        index = int(len(lines) * (4 / 5))
        convert_to_tsv(lines[:index], train_out)
        convert_to_tsv(lines[index:], test_out)
    else:
        out = open(output_file, 'w')
        convert_to_tsv(f, out)
