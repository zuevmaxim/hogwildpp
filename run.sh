#!/bin/bash

make clean
make

python3 collect_svm.py && echo "collect_svm OK"
python3 collect_numasvm.py && echo "collect_numasvm OK"

#sudo shutdown now -h
