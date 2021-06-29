#!/bin/bash

# runs preprocessing steps for a single given filepath, check if processed file already exists needs to be done by outer
# scripts that calls this bash scrip

# filepath needs to be specified as first positional cmd argument when calling this file
filename=$1

Intermediate_File_Directory='/net/n2o/wolke_scratch/kjeggle/IC_CIR/intermediate'
Template_Path='/home/kjeggle/cirrus/src/config_files/template.nc'

d=`echo $filename | grep -E -o '[0-9]{8}'`

echo "Start Horizontal remapping $d"

FINAL_FILE=${Intermediate_File_Directory}/remapped_${filename}.nc
if test -f "$FINAL_FILE"; then
  echo "Processed File already exists already exists."
  exit
fi

# 1. remap horizontally
cdo remapnn,$Template_Path $filename $FINAL_FILE
