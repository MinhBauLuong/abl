#!/bin/bash
#
# Helper script for noaa_snapshots.py
# Written by Eliot Quon (eliot.quon@nrel.gov)
#
# Before use, check the two overlapping time directories as well as the time offset below.
#
set -e
#time0=2016111206
#time1=2016111218
#offset=12 # hours

hours_offset()
{
    python -c "from datetime import datetime; DTobj = lambda s: datetime(int(s[:4]),int(s[4:6]),int(s[6:8]),int(s[8:10])); print int((DTobj('$2')-DTobj('$1')).seconds/3600.)"
}

if [ -z "$3" ]; then
    #echo 'Specify plot to merge'
    echo "USAGE: $0 [fieldName] [time0] [time1]"
else
    qty="$1"
    time0="$2"
    time1="$3"
    if [ -z "$4" ]; then
        #offset=$((time1-time0))
        offset=`hours_offset $time0 $time1`
        echo "Detected offset: $offset hours"
    fi
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
