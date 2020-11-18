#!/bin/bash
module load nco
for year in {1979..2017}
do
    echo $year
    ncks -d latitude,-15.0,15.0 /mnt/cmip5-data/reanalysis/era.interim/sfc/ttr/oper/ttr.$year.nc /climodes/data4/kcarr/era-interim/ttr.$year.nc
done