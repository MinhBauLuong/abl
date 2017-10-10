#!/usr/bin/env python
#
# Pulls HRRR simulation snapshots from noaa.gov
# E.g., images from:
# https://rapidrefresh.noaa.gov/hrrr/HRRRwfipsubh/Welcome.cgi?dsKey=hrrr_wfip_nest_subh_jet&domain=nest&run_time=12+Nov+2016+-+06Z
#
# Written by Eliot Quon (eliot.quon@nrel.gov) -- 2017-09-13
#
# Sample usage
# ------------
#   noaa_snapshots.py -p ustar -t 2016111206 
#   noaa_snapshots.py -p ustar -t 2016111218 
#   merge_snapshot_dirs.sh ustar
#
# This example will create time directories '201611206' and '2016111218' with a series of simualted
#   HRR data. The helper shell script will create a directory called 'ustar' and created symlinks to
#   the snapshots in the separate time directories, so that the complete time series can be
#   previewed in flipbook fashion.
#
import os
import argparse
import urllib2

# Sample complete URL:
#urlstr = 'https://rapidrefresh.noaa.gov/hrrr/HRRRwfipsubh/displayMapLocalDiskDateDomainZipTZA.cgi?keys=hrrr_wfip_nest_subh_jet:&runtime=2016111206&plot_type=sfc_ri&fcst=0100&time_inc=15&num_times=61&model=hrrr&ptitle=HRRR-WFIP2%20Model%20Fields%20-%20Experimental&maxFcstLen=15&fcstStrLen=-1&domain=nest&adtfn=1'

cgiurl = 'https://rapidrefresh.noaa.gov/hrrr/HRRRwfipsubh/displayMapLocalDiskDateDomainZipTZA.cgi?'
baseurl = cgiurl[:cgiurl.rfind('/')]

# defaults:
keys = 'hrrr_wfip_nest_subh_jet:'
ptitle = 'HRRR-WFIP2%20Model%20Fields%20-%20Experimental'
fcstStrLen = '-1'
num_times = '61'
adtfn = '1'


def get_imagepath(url):
    htmlfile = urllib2.urlopen(url)
    for line in htmlfile:
        if 'for_web' in line: break
    ist = line.find('src=')
    line = line[ist+5:].split('"')[0]
    return baseurl + '/' + line

def download_image(url,name=None):
    imgfile = urllib2.urlopen(url)
    if name is None:
        name = url.split('/')[-1]
    outfile = os.path.join(outdir,name)
    with open(outfile,'wb') as f:
        f.write(imgfile.read())


################################################################################
################################################################################
################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description='Tool for downloading a series of simulation snapshots from rapidrefresh.noaa.gov',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--plot', dest='plot_type', type=str, required=True,
            help='abbreviated plot name to retrieve (sfc_ri, sfc_sh, ustar, pblh, 80m_tke, ...)')
    parser.add_argument('-t', '--runtime', dest='runtime', type=str, required=True,
            default='2016111206', metavar='YYYYMMDDHH',
            help='date-time string for when the simulation was initialized; HH==(06,18)')
    parser.add_argument('-m', '--model', dest='model', type=str,
            default='hrrr',
            help='name of rapid refresh model')
    parser.add_argument('-d', '--domain', dest='domain', type=str,
            default='nest',
            help='name of computational domain')
    parser.add_argument('--max', dest='hrsmax', type=int,
            default=15,
            help='number of hours for which the simulation was run')

    # process inputs

    args = parser.parse_args()

    time_inc = str(args.hrsmax)
    maxFcstLen = time_inc
    hrs_range = range(1,args.hrsmax+1)

    outdir = os.path.join(args.runtime, args.plot_type)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # now actually generate URLs and download images

    for hr in hrs_range:
        tstr = '{:02d}00'.format(hr)
        newurlstr = cgiurl + \
            'keys='+keys + '&' \
            'runtime='+args.runtime + '&' \
            'plot_type='+args.plot_type + '&' \
            'fcst='+tstr + '&' \
            'time_inc='+time_inc + '&' \
            'num_times='+num_times + '&' \
            'model='+args.model + '&' \
            'ptitle='+ptitle + '&' \
            'maxFcstLen='+maxFcstLen + '&' \
            'fcstStrLen='+fcstStrLen + '&' \
            'domain='+args.domain + '&' \
            'adtfn='+adtfn
        imgurl = get_imagepath(newurlstr)
        print 'Downloading from',imgurl
        download_image(imgurl)

