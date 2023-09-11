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

def return_prefix(filename):
    
    # Use a function to regex the Day number and use that to sort
    day = findall('.(\d+)\. ',filename)
    assert(len(day) == 1)
    
    return int(day[0])

def parse_aligned_timecourse_directory(dirname,folder_str='*. Day*/',INCLUDE_ZERO=True):
    # Given a directory (of Prairie Instruments time course)
    # 
    
    filelist = pd.DataFrame()
    
    filelist['B'] = sorted(glob(path.join(dirname,folder_str, 'B_align.tif')), key = return_prefix)
    filelist['G'] = sorted(glob(path.join(dirname,folder_str, 'G_align.tif')), key = return_prefix)
    filelist['R'] = sorted(glob(path.join(dirname,folder_str, 'R_align.tif')), key = return_prefix)
    filelist['R_shg'] = sorted(glob(path.join(dirname,folder_str, 'R_shg_align.tif')), key = return_prefix)
    T = len(filelist)

    
    if INCLUDE_ZERO:
        # t= 0 has no '_align'imp
        s = pd.DataFrame({'B': sorted(glob(path.join(dirname,folder_str, 'B_reg.tif')))[0],
                          'G': sorted(glob(path.join(dirname,folder_str, 'G_reg.tif')))[0],
                          'R': sorted(glob(path.join(dirname,folder_str, 'R_reg_reg.tif')))[0],
                      'R_shg': sorted(glob(path.join(dirname,folder_str, 'R_shg_reg_reg.tif')))[0]},
                         index=[0])
        filelist.index = np.arange(1,T+1)
        filelist = pd.concat((s,filelist))
        filelist = filelist.sort_index()

    
    # heightmaps = sorted(glob(path.join(dirname,'*/heightmap.tif')),key=sort_by_day)

    # if len(heightmaps) == len(filelist):
    #     filelist['Heightmap'] = heightmaps
    
    return filelist


def parse_unreigstered_channels(dirname,folder_str='*. Day*/',sort_func=return_prefix):
    # Given a directory (of Prairie Instruments time course), grab all the _reg.tifs
    # (channels are not registered to each other)
    # 
    
    
    B = glob(path.join(dirname,folder_str, 'B_reg.tif'))
    
    idx = [return_prefix(f) for f in B]
    filelist = pd.DataFrame(index=idx)
    filelist.loc[idx,'B'] = B
    
    # filelist['G'] = sorted(glob(path.join(dirname,folder_str, 'G_reg.tif')), key = return_prefix)
    G = glob(path.join(dirname,folder_str, 'G_reg.tif'))
    idx = [return_prefix(f) for f in G]
    filelist.loc[idx,'G'] = B
    
    R = glob(path.join(dirname,folder_str, 'R_reg.tif'))
    idx = [return_prefix(f) for f in R]
    filelist.loc[idx,'R'] = R
    
    R_shg = glob(path.join(dirname,folder_str, 'R_shg_reg.tif'))
    idx = [return_prefix(f) for f in R_shg]
    filelist.loc[idx,'R_shg'] = R_shg
    
    filelist = filelist.sort_index()
    
    return filelist
    
def parse_unaligned_channels(dirname,folder_str='*. Day*/'):
    # Given a directory (of Prairie Instruments time course)
    # 
    
    # filelist['B'] = sorted(glob(path.join(dirname,folder_str, 'B_reg.tif')), key = return_prefix)
    B = glob(path.join(dirname,folder_str, 'B_reg.tif'))
    idx = [return_prefix(f) for f in B]
    filelist = pd.DataFrame(index=idx)
    filelist.loc[idx,'B'] = B
    
    # filelist['G'] = sorted(glob(path.join(dirname,folder_str, 'G_reg.tif')), key = return_prefix)
    G = glob(path.join(dirname,folder_str, 'G_reg.tif'))
    idx = [return_prefix(f) for f in G]
    filelist.loc[idx,'G'] = B
    
    R = glob(path.join(dirname,folder_str, 'R_reg_reg.tif'))
    idx = [return_prefix(f) for f in R]
    filelist.loc[idx,'R'] = R
    
    R_shg = glob(path.join(dirname,folder_str, 'R_shg_reg_reg.tif'))
    idx = [return_prefix(f) for f in R_shg]
    filelist.loc[idx,'R_shg'] = R_shg
    
    filelist = filelist.sort_index()
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
    
