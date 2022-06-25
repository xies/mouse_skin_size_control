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
import seaborn as sb
from re import match

# import sys; sys.path.insert(0,'/Users/xies/Code/xies_utils/basic_utils.py')
# from basic_utils import *

dirnames = {}
dirnames['WT R2'] = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/05-08-2022/F2 WT/R2/manual_track'
dirnames['KO R2'] = '/Users/xies/Desktop/KO R2/manual_track'

dx = 0.292435307476612

time_stamps = {}
time_stamps['WT R2'] = [0,0.5,1,1.5,2,2.5,3,3.5,4.5,5,5.5,6,6.5,7]
time_stamps['KO R2'] = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,
                        7.5,8,8.5,9,9.5,10]

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

for name,dirname in dirnames.items():
    
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
            birth_time = time_stamps[name][birth_frame - 1] * 24
            birth_size = birth['Volume'].values[0]
            
            # Division (may be missing)
            if (this_cell['State'] == 'd').sum() > 0:
                division = this_cell[ this_cell['State'] == 'd']
                division_frame = int(division['Frame'].values)
                division_time = time_stamps[name][division_frame - 1] * 24
                division_size = division['Volume'].values[0]
            elif (this_cell['State'] == 'sd').sum() > 0:
                division = this_cell[ this_cell['State'] == 'sd']
                division_frame = int(division['Frame'].values)
                division_time = time_stamps[name][division_frame - 1] * 24
                division_size = division['Volume'].values[0]
            else:
                division_frame = np.nan
                division_time = np.nan
                division_size = np.nan
            
            # S phase (may be missing)
            if (this_cell['State'] == 's').sum() > 0:
                sphase = this_cell[ this_cell['State'] == 's']
                sphase_frame = int(sphase['Frame'].values)
                sphase_time = time_stamps[name][sphase_frame - 1] * 24
                sphase_size = sphase['Volume'].values[0]
            elif (this_cell['State'] == 'sd').sum() > 0:
                sphase = this_cell[ this_cell['State'] == 'sd']
                sphase_frame = int(sphase['Frame'].values)
                sphase_time = time_stamps[name][sphase_frame] * 24
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
            
            # Ignore when birth or S phase is denoted at frame 8, since the time before this was lost.
            # Division still OK since division annotation depends on the subsequent frame, not before.
            ignore = False
            if genotype == 'WT':
                if sphase_frame == 8:
                    ignore = True
                elif birth_frame == 8:
                    ignore = True
            elif genotype =='KO':
                if sphase_frame == 15:
                    ignore = True
                elif birth_frame == 15:
                    ignore = True
            
            cell = pd.Series({'CellID': cellID
                              ,'LineageID':lineageID
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
                                   ,'Ignore': ignore
                                   })
            
            df = df.append(cell, ignore_index=True)
    

df_raw = df

df = df[df['Ignore'] == False]
ko = df[df['Genotype'] == 'KO']
wt = df[df['Genotype'] == 'WT']

# Check for duplicated CellIDs (manually done so there maybe some)
print('------\n Duplicated CellIDs:')
print(wt[wt.duplicated('CellID')])
print(ko[ko.duplicated('CellID')])

#%%
df_ = ko

#%% Some quality control plots. Some time frames are not as good as others

plt.figure()
plt.scatter(df_['Birth frame'],df_['Birth size'])
plt.figure()
plt.scatter(df_['S phase frame'],df_['S phase size'])
plt.figure()
plt.scatter(df_['Division frame'],df_['Division size'])

#%%

plt.figure()
plt.scatter(df_['Birth size'],df_['G1 growth'])
plt.scatter(df_['Birth size'],df_['Total growth'])
plt.legend(['G1 growth','Total growth'])

plt.figure()
plt.scatter(df_['Birth size'],df_['G1 length'])
plt.scatter(df_['Birth size'],df_['Cycle length'])
plt.legend(['G1 length','Total length'])


#%% Print stats

print('--- Growth, correlation')

X,Y = nonan_pairs(df_['Birth size'],df_['G1 growth'])
R,_ = np.corrcoef(X,Y)
print(f'Pearson R, x = birth size, y = g1 growth, R = {R[1]}')

X,Y = nonan_pairs(df_['Birth size'],df_['Total growth'])
R,_ = np.corrcoef(X,Y)
print(f'Pearson R, x = birth size, y = total growth, R = {R[1]}')


print('--- Time, correlation')
      
X,Y = nonan_pairs(df_['Birth size'],df_['G1 length'])
R,_ = np.corrcoef(X,Y)
print(f'Pearson R, x = birth size, y = g1 length, R = {R[1]}')

X,Y = nonan_pairs(df_['Birth size'],df_['Cycle length'])
R,_ = np.corrcoef(X,Y)
print(f'Pearson R, x = birth size, y = cycle length, R = {R[1]}')


print('--- Growth, regression')
      
X,Y = nonan_pairs(df_['Birth size'],df_['G1 growth'])
p = np.polyfit(X,Y,1)
print(f'Regression slope, x = birth size, y = g1 growth, m = {p[0]}')

X,Y = nonan_pairs(df_['Birth size'],df_['Total growth'])
p = np.polyfit(X,Y,1)
print(f'Regression slope, x = birth size, y = Total growth, m = {p[0]}')






