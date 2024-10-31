#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:26:03 2024

@author: xies
"""

import numpy as np
from glob import glob
from os import path
from re import findall
import xml.etree.ElementTree as ET
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd
from natsort import natsorted

dirname = '/Users/xies/OneDrive - Stanford/In vitro/CV from snapshot/zebrafish_ditalia/osx_fucci_26hpp_11_4_17/'

#%%

'''
This is a bit complicated...

Example XML: osx_fucci_26hpp_11_4_17_pt6_Mark_and_Find_002_Position001
Example tiff: osx_fucci_26hpp_11_4_17_pt4_Mark_and_Find_001_Position001_t2_ch00_stack.tiff

But at this 'experiment'/container, there could be multiple actual time points

Strategy: grab all the filenames from the actual stacks, parse the filename, and reverse engineer XML file
to get correct time stamp

'''

#%% Parse XML manifest

def parse_xml_name(f):
    part = findall('pt(\d+)',f)
    find = findall('Mark_and_Find_(\d+)',f)
    position = findall('Position(\d+)',f)
    assert len(part) == 1 and len(find) == 1 and len(position) == 1
    
    return part[0], find[0], position[0]


xmls = natsorted(glob(path.join(dirname,'MetaData/*Position002_Properties.xml')))

xml_manifest = pd.DataFrame(list(map(parse_xml_name,xmls)),columns=['Part','Find','Position'])
xml_manifest['xml'] = xmls

### Parse XML to get X/T dimensions

for idx,row in xml_manifest.iterrows():
    root = ET.parse(row['xml']).getroot()
    dimension_desc = [e for e in root.iter('DimensionDescription')]
    for dim in dimension_desc:
        if dim.attrib['DimID'] == 'T':
            T = int(dim.attrib['NumberOfElements'])
        elif dim.attrib['DimID'] == 'Z':
            Z = int(dim.attrib['NumberOfElements'])
    xml_manifest.at[idx,'Z'] = Z
    xml_manifest.at[idx,'T'] = T

xml_manifest.index = xml_manifest['xml']

#%% Parse XML time stamps (put into dict keyed by xml name) and use the manifest to look put where to put the info

def parse_datetime_from_leicaformat(date,time):
    # Strip the msec mark
    d = datetime.strptime(date,'%m/%d/%Y')
    t = datetime.strptime(time,'%I:%M:%S %p').time()
    return datetime.combine(d,t)

timestamps = {}
for idx,row in xml_manifest.iterrows():
    # start_time,end_time = parse_time_stamp_from_xml(xml_manifest.iloc[0]['xml'])
    with open(row['xml']) as f:
        soup = BeautifulSoup(f,'xml')
    
    timestamps_for_each_timepoint = soup.find_all('TimeStamp')[::4][:: int(row['Z'])]
    this_timestamps = []
    for t in timestamps_for_each_timepoint:
        this_timestamps.append(parse_datetime_from_leicaformat(date = t.attrs['Date'],time = t.attrs['Time']))
        
    if not len(timestamps_for_each_timepoint) == row['T']:
        missing_ts = int(row['T'] - len(timestamps_for_each_timepoint))
        for i in range(missing_ts):
            this_timestamps.append(np.nan)
    
    timestamps[idx] = this_timestamps

#%% Merge back onto TIFF manifest

img_files = natsorted(glob(path.join(dirname,'stacks/*Position002*ch00*.tif')))

def parse_img_name(f):
    part = findall('pt(\d+)',f)
    find = findall('Mark_and_Find_(\d+)',f)
    time = findall('_t(\d+)_',f)
    channel = findall('_ch(\d+)_',f)
    position = findall('Position(\d+)',f)
    assert len(part) == 1 and len(find) == 1 and len(time) == 1 and len(channel) == 1 and len(position) == 1
    
    return part[0], find[0], position[0], time[0], channel[0]

img_manifest = pd.DataFrame(list(map(parse_img_name, img_files)),columns=['Part','Find','Position','Time','Channel'])
img_manifest['img'] = img_files

manifest = pd.merge(img_manifest,xml_manifest,on=['Part','Find','Position'])

#%% Iterate through rows and use each time point to XML to index into dictionary, and pull out the right order of frame

for idx,row in manifest.iterrows():
    
    t = timestamps[row['xml']][int(row.Time)]
    manifest.at[idx,'Timestamp'] = t
    
manifest['Elapsed'] = manifest['Timestamp'] - manifest.iloc[0]['Timestamp']
manifest.to_csv(path.join(dirname,'Position002_manifest.csv'))
manifest.to_pickle(path.join(dirname,'Position002_manifest.pkl'))
    



