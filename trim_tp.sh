#!/bin/bash
module load nco

echo Getting tropical precip...
for year in {1979..2017}
do
    echo $year
    ncks -d latitude,-15.0,15.0 /mnt/cmip5-data/reanalysis/era.interim/sfc/tp/oper/tp.$year.nc /climodes/data4/kcarr/era-interim/tp.$year.nc
done

echo Getting U.S. precipitation...
for year in {1979..2017}
do
    echo $year
    ncks -d latitude,25.0,50.0 -d longitude,230,300 /mnt/cmip5-data/reanalysis/era.interim/sfc/tp/oper/tp.$year.nc /climodes/data4/kcarr/era-interim/tp_us.$year.nc
done