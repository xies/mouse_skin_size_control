#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 18:13:36 2022

@author: xies
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from skimage import io
from os import path, stat
from glob import glob
from tqdm import tqdm
import pandas as pd
import seaborn as sb
from re import match
import matplotlib.pylab as plt

# Custom packages
from basicUtils import nonans, nonan_pairs


dirnames = {}

dirnames['WT R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/06-25-2022/M1 WT/R1/manual_track'
dirnames['KO R1'] = '/Users/xies//OneDrive - Stanford/Skin/Two photon/NMS/06-25-2022/M6 RBKO/R1/manual_track'
dirnames['WT R2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/05-08-2022/F2 WT/R2/manual_track'
dirnames['KO R2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/05-08-2022/F1 RB-KO/R2/manual_track'

dx = {}
dx['WT R1'] = 0.292435307476612 /1.5
dx['KO R1'] = 0.292435307476612 /1.5
dx['WT R2'] = 0.292435307476612
dx['KO R2'] = 0.292435307476612

time_stamps = {}
time_stamps['WT R1'] = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,
                        7.5,8,8.5,9,9.5]
time_stamps['KO R1'] = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,
                        7.5,8,8.5,9,9.5]
time_stamps['WT R2'] = [0,0.5,1,1.5,2,2.5,3,3.5,4.5,5,5.5,6,6.5,7]
time_stamps['KO R2'] = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,
                        7.5,8,8.5,9,9.5,10]
mouse = {}
mouse['WT R1'] = 1
mouse['KO R1'] = 1
mouse['WT R2'] = 2
mouse['KO R2'] = 2

#%% Parse .csv files
'''

Parse everything into cell-centric dataframe

Folder structure

../LineageID/LineageID.MotherID.CellID/t{frame}.csv
../LineageID/LineageID.MotherID.CellID/t{frame}.b.csv -- birth
../LineageID/LineageID.MotherID.CellID/t{frame}.d.csv -- division
../LineageID/LineageID.MotherID.CellID/t{frame}.s.csv -- s phase entry

'''

df = pd.DataFrame()

def sort_by_last_field(cellID):
    return int(cellID.split('.')[2])

for name,dirname in dirnames.items():
    
    print(f'{dirname}')
    all_cells = sorted(glob(path.join(dirname,'*/*')),key = sort_by_last_field)
    
    genotype = name.split(' ')[0]
    
    for f in tqdm(all_cells):
        # Parse the manual annotations
        _tmp = pd.DataFrame()
        
        # Parse lineage annotation
        subdir = path.split(f)[1]
        lineageID, motherID, cellID = subdir.split('.')
        uniqueID = subdir

        if motherID == '_':
            motherID = np.nan
        else:
            motherID = int(motherID)
        
        time_points = glob(path.join(f,'t*.csv'))
        
        track = pd.DataFrame()
        for i, this_time in enumerate(time_points):
            
            fname = path.basename(this_time)

            # Parse cell cycle annotation (if applicable)
            basename = path.basename(this_time)

            [frame,state,_] = fname.split('.')
            frame = int(frame[1:])
            
            _x = pd.read_csv(this_time)
            if not 'Area' in _x.columns:
                _x = pd.read_csv(this_time,delimiter='\t')
                
            volume = _x['Area'].sum() * dx[name] ** 2
            
            _tmp.at[i,'CellID'] = float(cellID)
            _tmp.at[i,'State'] = state
            _tmp.at[i,'Frame'] = int(frame)
            _tmp.at[i,'Volume'] = volume
            
            # _tmp.at[i,'X'] = _x['X'].mean()
            # _tmp.at[i,'Y'] = _x['Y'].mean()
    
        # Multiple cellIDs could be in the same lineage folder
        cells_in_lineage = np.unique(_tmp['CellID'])
        for cellID in cells_in_lineage:
            
            this_cell = _tmp[_tmp['CellID'] == cellID]
            # x = this_cell.iloc[0]['X']; y = this_cell.iloc[0]['Y']
            
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
            
            cell = pd.Series({'CellID': cellID
                              ,'UniqueID':uniqueID
                              ,'LineageID':lineageID
                              ,'MotherID': motherID
                            ,'Genotype': genotype
                            ,'Region': name
                            ,'Mouse': mouse[name]
                            ,'Directory':dirname
                            # ,'X': x
                            # ,'Y': y
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
    
df_raw = df.copy()

#% Lineage calculations

def extrapolate_division_size(df):
    genotypes = np.unique(df['Genotype'])  

    df['Total daughter size'] = np.nan
    df['Division size (sm)'] = np.nan
    df['Daughter asymmetry'] = np.nan
    
    for i,cell in df.iterrows():
        Igeno = df['Genotype'] == cell['Genotype']
        
        # Find cell that has this cellID as motherID
        daughters = df[ (df['MotherID'] == cell['CellID']) & Igeno]
        if len(daughters) == 2:
            
            total = daughters['Birth size'].sum()
            df.at[i,'Total daughter size'] = total
            
            div_size_ext = (cell['Division size'] + total ) / 2
            df.at[i,'Division size (sm)'] = div_size_ext
            
            df.at[i,'Daughter asymmetry'] = \
                np.abs(daughters.iloc[0]['Birth size'] - daughters.iloc[1]['Birth size'])
            
        else:
            continue
        
    return df

df = extrapolate_division_size(df_raw)
df['Mother daughter diff'] = df['Total daughter size'] - df['Division size']
df['Fold grown'] = df['Division size'] / df['Birth size']
df['Fold grown (sm)'] = df['Division size (sm)'] / df['Birth size']
df['Total growth (sm)'] = df['Division size (sm)']  - df['Birth size']

I_ignore = (df['Genotype'] == 'WT') & ((df['Birth frame'] == 11) \
            | (df['Birth frame'] == 10))
df = df[~I_ignore]

ko = df[df['Genotype'] == 'KO']
wt = df[df['Genotype'] == 'WT']

# Check for duplicated CellIDs (manually done so there maybe some)
print('------\n Duplicated CellIDs:')
print(wt[wt.duplicated(subset = ['CellID','Directory'])])
print(ko[ko.duplicated(['CellID','Directory'])])

#%%

df_ = wt; title_str = 'WT'
# df_ = ko; title_str = 'RB-KO'



