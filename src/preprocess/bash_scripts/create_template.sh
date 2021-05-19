# creates template used for horizontal regridding
# global grid with 0.25x0.25 resolution

Grid_Spec_Path='/home/kjeggle/cirrus/src/config_files/gridspec'
Template_Path='/home/kjeggle/cirrus/src/config_files/gridspec/template.nc'

cdo -f nc -sellonlatbox,-180,180,-90,90 -random,r1440x720 $Template_Path # create target grid
cdo setgrid,${Grid_Spec_Path} $Template_Path $Template_Path # template grid has an offset for some reason, force correct starting point