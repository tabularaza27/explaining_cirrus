# creates template used for horizontal regridding

MIN_LON=-75
MAX_LON=-15
MIN_LAT=0
MAX_LAT=60

Grid_Spec_Path='/home/kjeggle/cirrus/src/config_files/gridspec'
Template_Path='/home/kjeggle/cirrus/src/config_files/gridspec/template.nc'

cdo -f nc -sellonlatbox,-180,180,-90,90 -random,r1440x720 $Template_Path # create target grid
cdo setgrid,${Grid_Spec_Path} $Template_Path $Template_Path # template grid has an offset for some reason, force correct starting point
cdo sellonlatbox,$MIN_LON,$MAX_LON,$MIN_LAT,$MAX_LAT $Template_Path $Template_Path