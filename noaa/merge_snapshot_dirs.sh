#!/bin/bash
#
# Helper script for noaa_snapshots.py
# Written by Eliot Quon (eliot.quon@nrel.gov)
#
# Before use, check the two overlapping time directories as well as the time offset below.
#
set -e
time0=2016111206
time1=2016111218
offset=12

if [ -z "$1" ]; then
    echo 'Specify plot to merge'
else
    qty="$1"
    mkdir -p $qty
    rootdir=`pwd`
    cd $qty
    for f in $rootdir/$time0/$qty/*; do
        name=${f##*/}
        name=${name%.png}
        tstr=${name:(-4)}
        tstr=${tstr:0:2} # read hour str only
        name=${name%??00}
        fname=$name$tstr.png
        ln -sv $f $fname
    done
    for f in $rootdir/$time1/$qty/*; do
        name=${f##*/}
        name=${name%.png}
        tstr=${name:(-4)}
        tstr=${tstr:0:2} # read hour str only
        tval=`echo $tstr | sed 's/^0*//'` # remove leading 0 if it's there so we can do some basic addition
        name=${name%??00}
        tstr=`printf '%02d' $((tval+offset))` # add offset to time string
        fname=$name$tstr.png
        if [ -f "$fname" ]; then
            fname="$name${tstr}a.png" # append 'a' suffix if file already exists
        fi
        ln -sv $f $fname
    done
fi
