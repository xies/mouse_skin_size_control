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
import matplotlib.pyplot as plt
import numpy as np

def sort_by_prefix(filename):
    
    # Use a function to regex the Day number and use that to sort
    day = findall('.(\d+)\. ',filename)
    assert(len(day) == 1)
    
    return int(day[0])

def parse_aligned_timecourse_directory(dirname,folder_str='*. Day*/',INCLUDE_ZERO=True):
    # Given a directory (of Prairie Instruments time course)
    # 
        
    filelist = pd.DataFrame()
    filelist['B'] = sorted(glob(path.join(dirname,folder_str, 'B_align.tif')), key = sort_by_prefix)
    filelist['G'] = sorted(glob(path.join(dirname,folder_str, 'G_align.tif')), key = sort_by_prefix)
    filelist['R'] = sorted(glob(path.join(dirname,folder_str, 'R_align.tif')), key = sort_by_prefix)
    filelist['R_shg'] = sorted(glob(path.join(dirname,folder_str, 'R_shg_align.tif')), key = sort_by_prefix)
    T = len(filelist)
    filelist.index = np.arange(1,T+1)
    
    if INCLUDE_ZERO:
        # t= 0 has no '_align'imp
        s = pd.DataFrame({'B': sorted(glob(path.join(dirname,folder_str, 'B_reg.tif')))[0],
                          'G': sorted(glob(path.join(dirname,folder_str, 'G_reg.tif')))[0],
                          'R': sorted(glob(path.join(dirname,folder_str, 'R_reg_reg.tif')))[0],
                      'R_shg': sorted(glob(path.join(dirname,folder_str, 'R_shg_reg_reg.tif')))[0]},
                         index=[0])

        filelist = pd.concat((s,filelist))
        filelist = filelist.sort_index()
    
    # heightmaps = sorted(glob(path.join(dirname,'*/heightmap.tif')),key=sort_by_day)

    # if len(heightmaps) == len(filelist):
    #     filelist['Heightmap'] = heightmaps
    
    return filelist


def parse_unreigstered_channels(dirname,folder_str='*. Day*/',sort_func=sort_by_prefix):
    # Given a directory (of Prairie Instruments time course), grab all the _reg.tifs
    # (channels are not registered to each other)
    # 
    
    print(path.join(dirname,folder_str + 'B_reg.tif'))
    filelist = pd.DataFrame()
    filelist['B'] = sorted(glob(path.join(dirname,folder_str, 'B_reg.tif')), key = sort_func)
    filelist['G'] = sorted(glob(path.join(dirname,folder_str, 'G_reg.tif')), key = sort_func)
    filelist['R'] = sorted(glob(path.join(dirname,folder_str, 'R_reg.tif')), key = sort_func)
    filelist['R_shg'] = sorted(glob(path.join(dirname,folder_str, 'R_shg_reg.tif')), key = sort_func)
    return filelist
    
def parse_unaligned_channels(dirname,folder_str='*. Day*/'):
    # Given a directory (of Prairie Instruments time course)
    # 
        
    filelist = pd.DataFrame()
    filelist['B'] = sorted(glob(path.join(dirname,folder_str, 'B_reg.tif')), key = sort_by_prefix)
    filelist['G'] = sorted(glob(path.join(dirname,folder_str, 'G_reg.tif')), key = sort_by_prefix)
    filelist['R'] = sorted(glob(path.join(dirname,folder_str, 'R_reg_reg.tif')), key = sort_by_prefix)
    filelist['R_shg'] = sorted(glob(path.join(dirname,folder_str, 'R_shg_reg_reg.tif')), key = sort_by_prefix)
    # T = len(filelist)
    
    return filelist

def plot_cell_volume(track,x='Frame',y='Volume'):
    t = track[x]
    y = track[y]
    if 'Mitosis' in track.columns:
        if track.iloc[0]['Mitosis']:
            t = t[:-1]
            y = y[:-1]
    plt.plot(t,y)
    
