#!/bin/bash

#tmux new -s caltra_$1

module load dyn_tools

export MODEL=era5
export NETCDF_FORMAT=CF
export LAGRANTO=${DYN_TOOLS}/lagranto.era5
model

# lagranto help page:
# dyn_help lagranto
# for particular calls:
# lagrantohelp create_startf

config_id=$1 # config_id needs to be specified as second positional cmd argument when calling this file
dat=$2 # date of start file

# input directory: ERA5 netcdf files
era5filedir=/net/thermo/atmosdyn/era5/cdf

# get directories for given config
outfiledir=`python -c "from src.scaffolding.scaffolding import get_data_product_dir; from src.preprocess.helpers.constants import BACKTRAJ_OUTFILES; dir=get_data_product_dir('${config_id}', BACKTRAJ_OUTFILES); print(dir)"`
startfiledir=`python -c "from src.scaffolding.scaffolding import get_data_product_dir; from src.preprocess.helpers.constants import BACKTRAJ_STARTFILES; dir=get_data_product_dir('${config_id}', BACKTRAJ_STARTFILES); print(dir)"`

#outfiledir=/net/n2o/wolke_scratch/kjeggle/BACKTRAJECTORIES/outfiles # move out files to north atlantic domain

# define start and end date
dat_yyyy=`echo $dat | cut -c 1-4`
dat_mm=`echo $dat | cut -c 5-6`
dat_dd=`echo $dat | cut -c 7-8`
backdat=$(newtime ${dat} -10)  # 2.5d backward

#cd /net/litho/atmosdyn/binderh/varia/kai/test

# target dir, caltra script will be executed here
# directory where netcdf files are linked and where trajectories will be written
target_dir=${outfiledir}/${dat_yyyy}/${dat_mm}/${dat_dd}
echo $target_dir
# create target dir
mkdir -p $target_dir
cd $target_dir

# link era5 files of the last 60 hours
for i in {0..11}
do
  hourbackdat=$(newtime ${dat} -$i) # subtract one hour
  yyyy=`echo $hourbackdat | cut -c 1-4`
  mm=`echo $hourbackdat | cut -c 5-6`
  dd=`echo $hourbackdat | cut -c 7-8`
  hh=`echo $hourbackdat | cut -c 10-11`
  ln -sf ${era5filedir}/${yyyy}/${mm}/*${yyyy}${mm}${dd}_${hh}* .
done

# link startfile to output dir
startf_file=startf_${dat}
ln -sf ${startfiledir}/${dat_yyyy}/${startf_file} .
echo $startf_file

#ln -s /net/thermo/atmosdyn/era5/cdf/2008/01/

# link era5 netcdf files to current working directory
# commented out → do for whole month instead of for every file

# convert pressure in startfile from Pa to hPa
#awk '{print $1,$2,$3/100}' startf_20080129_17_Pascal > startf_20080129_17

### calculate tra
caltra ${dat} ${backdat} $startf_file tra_tmp_${dat}.1 -j
#caltra ${dat} ${backdat} /home/binderh/prog/varia/kai/startf_20080129_17 tra_tmp_${dat}.1 -j

### add height to trajectories (in km)
/home/binderh/prog/programs/height.to.traj/z2traj tra_tmp_${dat}.1 trah_tmp_${dat}.1

### tra tracen
if [ ! -f tracevars ];then
    echo "T           1.    0    P" >> tracevars   # K
    echo "Q        1000.    0    P" >> tracevars   # g/kg
    echo "LWC   1000000.    0    P" >> tracevars   # mg/kg
    echo "RWC   1000000.    0    P" >> tracevars   # mg/kg
    echo "IWC   1000000.    0    P" >> tracevars   # mg/kg
    echo "SWC   1000000.    0    P" >> tracevars   # mg/kg
    echo "U           1.    0    P" >> tracevars   # m/s
    echo "V           1.    0    P" >> tracevars   # m/s
    echo "OMEGA       1.    0    P" >> tracevars   # Pa/s
#    echo "o3       1000.    0    O" >> tracevars   # g/kg
    echo "cc          1.    0    O" >> tracevars   # percentage
fi
trace trah_tmp_${dat}.1 tra_traced_${dat}.1
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