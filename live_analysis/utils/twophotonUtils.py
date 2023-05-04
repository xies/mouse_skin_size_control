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
from scipy.interpolate import UnivariateSpline

def sort_by_day(filename):

    # Use a function to regex the Day number and use that to sort
    day = findall('.(\d+)\. Day',filename)
    assert(len(day) == 1)
    return int(day[0])

def parse_aligned_timecourse_directory(dirname):
    # Given a directory (of Prairie Instruments time course)
    # 
        
    filelist = pd.DataFrame()
    filelist['B'] = sorted(glob(path.join(dirname,'*. Day*/B_align.tif')), key = sort_by_day)
    filelist['G'] = sorted(glob(path.join(dirname,'*. Day*/G_align.tif')), key = sort_by_day)
    filelist['R'] = sorted(glob(path.join(dirname,'*. Day*/R_align.tif')), key = sort_by_day)
    filelist['R_shg'] = sorted(glob(path.join(dirname,'*. Day*/R_shg_align.tif')), key = sort_by_day)
    T = len(filelist)
    filelist.index = np.arange(0,T)
    print(filelist)
    
    # t= 0 has no '_align'imp
    # print(glob(path.join(dirname,'0. Day */B_reg_reg.tif')))
    # s = pd.Series({'B': glob(path.join(dirname,'0. Day */B_reg.tif'))[0],
    #                  'G': glob(path.join(dirname,'0. Day */G_reg.tif'))[0],
    #                  'R': glob(path.join(dirname,'0. Day */R_reg_reg.tif'))[0],
    #               'R_shg': glob(path.join(dirname,'0. Day */R_shg_reg_reg.tif'))[0]},
    #               name=0)
    
    # filelist = filelist.append(s)
    # filelist = filelist.sort_index()
    
    heightmaps = sorted(glob(path.join(dirname,'*/heightmap.tif')),key=sort_by_day)

    if len(heightmaps) == len(filelist):
        filelist['Heightmap'] = heightmaps
    
    return filelist


def parse_unaligned_channels(dirname):
    # Given a directory (of Prairie Instruments time course)
    # 
        
    filelist = pd.DataFrame()
    filelist['B'] = sorted(glob(path.join(dirname,'*. Day*/B_reg_reg.tif')), key = sort_by_day)
    filelist['G'] = sorted(glob(path.join(dirname,'*. Day*/G_reg_reg.tif')), key = sort_by_day)
    filelist['R'] = sorted(glob(path.join(dirname,'*. Day*/R_reg_reg.tif')), key = sort_by_day)
    filelist['R_shg'] = sorted(glob(path.join(dirname,'*. Day*/R_shg_reg_reg.tif')), key = sort_by_day)
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
    
    

def smooth_growth_curve(cf,x='Age',y='Volume',smoothing_factor=1e10):

    X = cf[x]
    Y = cf[y]
    
    I = (~np.isnan(X)) * (~np.isnan(Y))
        
    # Won't smooth 3 pts or fewer (cubic spline)
    if len(X[I]) < 4:
        Yhat = cf[y].values
        
    else:

        
        # Spline smooth
        spl = UnivariateSpline(X[I], Y[I], k=3, s=smoothing_factor)
        Yhat = spl(X)
        
    return Yhat
    


    
    