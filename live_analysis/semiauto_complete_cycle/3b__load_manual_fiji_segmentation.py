#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 22:28:50 2022

@author: xies
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from skimage import io
from os import path, stat
from glob import glob
import pandas as pd
from re import match
import seaborn as sb


dirnames = {}
dirnames['/Users/xies/Box/Mouse/Skin/Two photon/NMS/05-08-2022/F2 WT/R2/manual_track'] = 'WT R2'

dx = 0.2920097

time_stamps = [0,0.5,1,1.5,2,2.5,3,3.5,4.5,5,5.5,6,6.5,7]

#%% Parse .csv files
'''

Parse everything into cell-centric dataframe

Folder structure

../LineageID/CellID.{frame}.csv
../LineageID/CellIDt{frame}.csv
../LineageID/CellID.t{frame}.b.csv -- birth
../LineageID/CellID.t{frame}.d.csv -- division
../LineageID/CellID.t{frame}.s.csv -- s phase entry

'''

df = pd.DataFrame()

for dirname,name in dirnames.items():
    
    print(f'{dirname}')
    all_cells = glob(path.join(dirname,'*'))
    
    genotype = name.split(' ')[0]
    
    for f in all_cells:
        # Parse the manual annotations
        _tmp = pd.DataFrame()
        
        # Parse lineage annotation
        lineageID = path.split(f)[1]
        
        time_points = glob(path.join(f,'*.*.csv'))
        
        track = pd.DataFrame()
        for i, this_time in enumerate(time_points):
            
            fname = path.basename(this_time)

            # Parse cell cycle annotation (if applicable)
            basename = path.basename(this_time)

            [cellID,frame,state,_] = fname.split('.')
            
            _x = pd.read_csv(this_time)
            if not 'Area' in _x.columns:
                _x = pd.read_csv(this_time,delimiter='\t')
                
            volume = _x['Area'].sum() * dx ** 2
            
            _tmp.at[i,'CellID'] = float(cellID)
            _tmp.at[i,'State'] = state
            _tmp.at[i,'Frame'] = int(frame)
            _tmp.at[i,'Volume'] = volume
    
        # Multiple cellIDs could be in the same lineage folder
        cells_in_lineage = np.unique(_tmp['CellID'])
        for cellID in cells_in_lineage:
            
            this_cell = _tmp[_tmp['CellID'] == cellID]
            
            # Birth (should always exist)
            birth = this_cell[this_cell['State'] == 'b']
            birth_frame = int(birth['Frame'].values)
            birth_time = time_stamps[birth_frame - 1] * 12
            birth_size = birth['Volume'].values[0]
            
            # Division (may be missing)
            if (this_cell['State'] == 'd').sum() > 0:
                division = this_cell[ this_cell['State'] == 'd']
                division_frame = int(division['Frame'].values)
                division_time = time_stamps[division_frame - 1] * 12
                division_size = division['Volume'].values[0]
            elif (this_cell['State'] == 'sd').sum() > 0:
                division = this_cell[ this_cell['State'] == 'sd']
                division_frame = int(division['Frame'].values)
                division_time = time_stamps[division_frame - 1] * 12
                division_size = division['Volume'].values[0]
            else:
                division_frame = np.nan
                division_time = np.nan
                division_size = np.nan
            
            # S phase (may be missing)
            if (this_cell['State'] == 's').sum() > 0:
                sphase = this_cell[ this_cell['State'] == 's']
                sphase_frame = int(sphase['Frame'].values)
                sphase_time = time_stamps[sphase_frame - 1] * 12
                sphase_size = sphase['Volume'].values[0]
            elif (this_cell['State'] == 'sd').sum() > 0:
                sphase = this_cell[ this_cell['State'] == 'sd']
                sphase_frame = int(sphase['Frame'].values)
                sphase_time = time_stamps[sphase_frame] * 12
                sphase_size = sphase['Volume'].values[0]
            else:
                sphase_frame = np.nan
                sphase_time = np.nan
                sphase_size = np.nan
                
            
            # Cell cycle durations
            cycle_length = division_time - birth_time
            g1_length = sphase_time - birth_time
            
            # Growth amount
            cycle_growth = division_size - birth_size
            g1_growth = sphase_size - birth_size
            
            cell = pd.Series({'CellID': cellID
                                   ,'Genotype': genotype
                                   ,'Region': name
                                   ,'Directory':dirname
                                   ,'Birth time':birth_time
                                   ,'Birth frame':birth_frame
                                   ,'Birth size':birth_size
                                   ,'Division time':division_time
                                   ,'Division frame':division_frame
                                   ,'Division size':division_size
                                   ,'S phase time':sphase_time
                                   ,'S phase frame':sphase_frame
                                   ,'S phase size':sphase_size
                                   ,'Cycle length':cycle_length
                                   ,'G1 length':g1_length
                                   ,'Total growth':cycle_growth
                                   ,'G1 growth':g1_growth
                                   })
            
            df = df.append(cell, ignore_index=True)
    

df['Ignore'] = False
df.at[df['S phase frame'] == 8,'Ignore'] = True
df.at[df['Birth frame'] == 8,'Ignore'] = True
df.at[df['Division frame'] == 8,'Ignore'] = True

df_ = df[df['Ignore'] == False]



