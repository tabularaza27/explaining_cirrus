#!/bin/bash

# creates new tmux session for caltra to run in. prevents caltra calculations to get into each others way

dat=$1

tmux new -d -s caltra_${dat} "bash calc_backtrajectories.sh ${dat}"
