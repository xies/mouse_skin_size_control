#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:26:03 2024

@author: xies
"""

from glob import glob
from os import path
from re import findall
# import xml.etree.ElementTree as ET
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

img_files = natsorted(glob(path.join(dirname,'stacks/*Position001*ch00*.tif')))

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

#%% Map back the corresponding XML file using table joins

def parse_xml_name(f):
    part = findall('pt(\d+)',f)
    find = findall('Mark_and_Find_(\d+)',f)
    position = findall('Position(\d+)',f)
    assert len(part) == 1 and len(find) == 1 and len(position) == 1
    
    return part[0], find[0], position[0]


xmls = natsorted(glob(path.join(dirname,'MetaData/*Position001_Properties.xml')))

xml_manifest = pd.DataFrame(list(map(parse_xml_name,xmls)),columns=['Part','Find','Position'])
xml_manifest['xml'] = xmls

manifest = pd.merge(img_manifest,xml_manifest,on=['Part','Find','Position'])

#%% Parse XML time stamps (put into dict keyed by xml name) and use the manifest to look put where to put the info

from datetime import datetime

def parse_time_stamp_from_xml(filename):
    with open(filename) as f:
        soup = BeautifulSoup(f,'xml')
        
    start_time = list(soup.find_all('StartTime')[0].children)[0]
    # Strip the msec mark
    start_time = findall('(.+)\.',start_time)[0]
    start_time = datetime.strptime(start_time,'%m/%d/%Y %I:%M:%S %p')
    
    end_time = list(soup.find_all('EndTime')[0].children)[0]
    # Strip the msec mark
    end_time = findall('(.+)\.',start_time)[0]
    end_time = datetime.strptime(start_time,'%m/%d/%Y %I:%M:%S %p')
    
    return start_time,end_time

start_time,end_time = parse_time_stamp_from_xml(xml_manifest.iloc[0]['xml'].values)
    

