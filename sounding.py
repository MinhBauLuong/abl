#
# module to process sounding data
#
# written by Eliot Quon (eliot.quon@nrel.gov)
#
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup # for HTML parsing

def get_sounding(url):
    """Grab 'list' data from http://weather.uwyo.edu/upperair/sounding.html"""
    req = urlopen(url)
    output = BeautifulSoup(req,features='html.parser')
    obs,info = [],[]
    for pre in output.find_all('pre'):
        text = pre.text.strip()
        if text.startswith('---'):
            df, columns, units = _read_sounding_obs(text)
            heights = df.loc[~pd.isna(df['TEMP']),'HGHT']
            df['height_AGL'] = df['HGHT'] - heights.iloc[0]
            obs.append(df)
        else:
            info.append(_read_sounding_info(text))
    assert(len(obs)==len(info))
    for i in range(len(obs)):
        station = info[i]['Station identifier']
        datetime = pd.to_datetime(info[i]['Observation time'],
                                  format='%y%m%d/%H%M')
        obs[i]['datetime'] = datetime
        obs[i]['station'] = station
        obs[i] = obs[i][['datetime','station','height_AGL']+columns]
        info[i]['Units'] = units
    return obs, info

def _read_sounding_obs(text):
    """Read observations from station"""
    lines = text.split('\n')
    assert(lines[0].startswith('---'))
    columns = lines[1].split()
    units = lines[2].split()
    assert(len(columns)==len(units))
    assert(lines[3].startswith('---'))
    buf = StringIO(text)
    df = pd.read_fwf(buf, names=columns, skiprows=4)
    return df, columns, units

def _read_sounding_info(text):
    """Read station information and sounding indices"""
    info = {}
    for line in text.split('\n'):
        line = line.split(':')
        key = line[0].strip()
        val = line[1].strip()
        info[key] = val
    return info
