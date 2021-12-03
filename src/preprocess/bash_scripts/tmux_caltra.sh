#!/bin/bash

# creates new tmux session for caltra to run in. prevents caltra calculations to get into each others way

dat=$1

tmux new -d -s caltra_${dat} "bash /net/n2o/wolke/kjeggle/Repos/cirrus/src/preprocess/bash_scripts/calc_backtrajectories.sh ${dat}"
