#!/bin/bash

module load dyn_tools

export MODEL=era5
export NETCDF_FORMAT=CF
export LAGRANTO=${DYN_TOOLS}/lagranto.era5
model

# lagranto help page:
# dyn_help lagranto
# for particular calls:
# lagrantohelp create_startf

# input directory: ERA5 netcdf files
era5filedir=/net/thermo/atmosdyn/era5/cdf

# output directory
outfiledir=/net/n2o/wolke_scratch/kjeggle/BACKTRAJECTORIES/outfiles

# define start and end date
dat=$1
backdat=$(newtime ${dat} -60)  # 2.5d backward

#cd /net/litho/atmosdyn/binderh/varia/kai/test

# target dir, caltra script will be executed here
# directory where netcdf files are linked and where trajectories will be written
target_dir=${outfiledir}/${yyyy}/${mm}/${dd}

# create target dir
mkdir -p $target_dir
cd $target_dir

# link era5 files of the last 60 hours
for i in {1..61}
do
  hourbackdat=$(newtime ${dat} -$i) # subtract one hour
  yyyy=`echo $hourbackdat | cut -c 1-4`
  mm=`echo $hourbackdat | cut -c 5-6`
  dd=`echo $hourbackdat | cut -c 7-8`
  hh=`echo $hourbackdat | cut -c 10-11`
  ln -sf /net/thermo/atmosdyn/era5/cdf/${yyyy}/${mm}/*${yyyy}${mm}${dd}_${hh} .
done

#ln -s /net/thermo/atmosdyn/era5/cdf/2008/01/

# link era5 netcdf files to current working directory
# commented out → do for whole month instead of for every file

# define specs of start date
yyyy=`echo $dat | cut -c 1-4`
mm=`echo $dat | cut -c 5-6`
dd=`echo $dat | cut -c 7-8`


# convert pressure in startfile from Pa to hPa
#awk '{print $1,$2,$3/100}' startf_20080129_17_Pascal > startf_20080129_17

### calculate tra
startf_file=start_files/${yyyy}/startf_${dat}
echo $startf_file
#caltra ${dat} ${backdat} $startf_file tra_tmp_${dat}.1 -j
##caltra ${dat} ${backdat} /home/binderh/prog/varia/kai/startf_20080129_17 tra_tmp_${dat}.1 -j
#
#### add height to trajectories (in km)
#/home/binderh/prog/programs/height.to.traj/z2traj tra_tmp_${dat}.1 trah_tmp_${dat}.1
#
#### tra tracen
#if [ ! -f tracevars ];then
#    echo "T           1.    0    P" >> tracevars   # K
#    echo "Q        1000.    0    P" >> tracevars   # g/kg
#    echo "LWC   1000000.    0    P" >> tracevars   # mg/kg
#    echo "RWC   1000000.    0    P" >> tracevars   # mg/kg
#    echo "IWC   1000000.    0    P" >> tracevars   # mg/kg
#    echo "SWC   1000000.    0    P" >> tracevars   # mg/kg
#    echo "U           1.    0    P" >> tracevars   # m/s
#    echo "V           1.    0    P" >> tracevars   # m/s
#    echo "OMEGA       1.    0    P" >> tracevars   # Pa/s
#fi
#trace trah_tmp_${dat}.1 tra_traced_${dat}.1
#
## append RHi
##dim=`${LAGRANTO}/goodies/trainfo.sh ${tra_traced_${dat}.1} dim` #  dimensions of the trajectory file: #tra, #ntimes, #ncolumns
##\rm -f rhi_to_tra.param
##echo \"${inpfile}\"   > rhi_to_tra.param
##echo \"${outfile}\"  >> rhi_to_tra.param
##echo ${dim}          >> rhi_to_tra.param
#
#
#### clean
#rm tra_tmp_${dat}.1 trah_tmp_${dat}.1 # startf_${dat}.1