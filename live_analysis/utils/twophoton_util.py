#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:59:29 2022

@author: xies
"""

from glob import glob
import pandas as pd
from re import findall
from os import path
import numpy as np

def sort_by_day(filename):

    # Use a function to regex the Day number and use that to sort
    day = findall('.(\d+)\. Day',filename)
    assert(len(day) == 1)
    return int(day[0])

def parse_timecourse_directory(dirname):
    # Given a directory (of Prairie Instruments time course)
    # 
        
    filelist = pd.DataFrame()
    filelist['B'] = sorted(glob(path.join(dirname,'*. Day*/ZSeries*/B_align.tif')), key = sort_by_day)
    filelist['G'] = sorted(glob(path.join(dirname,'*. Day*/ZSeries*/G_align.tif')), key = sort_by_day)
    filelist['R'] = sorted(glob(path.join(dirname,'*. Day*/ZSeries*/R_align.tif')), key = sort_by_day)
    filelist['R_shg'] = sorted(glob(path.join(dirname,'*. Day*/ZSeries*/R_shg_align.tif')), key = sort_by_day)
    T = len(filelist)
    filelist.index = np.arange(1,T+1)
    
    # t= 0 has no '_align'imp
    s = pd.Series({'B': glob(path.join(dirname,'0. Day */ZSeries*/B_reg_reg.tif'))[0],
                     'G': glob(path.join(dirname,'0. Day */ZSeries*/G_reg_reg.tif'))[0],
                     'R': glob(path.join(dirname,'0. Day */ZSeries*/R_reg_reg.tif'))[0],
                  'R_shg': glob(path.join(dirname,'0. Day */ZSeries*/R_shg_reg_reg.tif'))[0]},
                  name=0)
    
    filelist = filelist.append(s)
    filelist = filelist.sort_index()
    
    heightmaps = sorted(glob(path.join(dirname,'*/heightmap.tif')),key=sort_by_day)

    if len(heightmaps) == len(filelist):
        filelist['Heightmap'] = heightmaps
    
    return filelist


def parse_unaligned_channels(dirname):
    # Given a directory (of Prairie Instruments time course)
    # 
        
    filelist = pd.DataFrame()
    filelist['B'] = sorted(glob(path.join(dirname,'*. Day*/ZSeries*/B_reg_reg.tif')), key = sort_by_day)
    filelist['G'] = sorted(glob(path.join(dirname,'*. Day*/ZSeries*/G_reg_reg.tif')), key = sort_by_day)
    filelist['R'] = sorted(glob(path.join(dirname,'*. Day*/ZSeries*/R_reg_reg.tif')), key = sort_by_day)
    filelist['R_shg'] = sorted(glob(path.join(dirname,'*. Day*/ZSeries*/R_shg_reg_reg.tif')), key = sort_by_day)
    T = len(filelist)
    
    return filelist