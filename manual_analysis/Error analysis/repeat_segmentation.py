#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:19:21 2019

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os.path as path
import pickle as pkl
from scipy import stats
from glob import glob

#Load df from pickle
r1 = pd.read_pickle('/Users/xies/Box/Mouse/Skin/W-R1/tracked_cells/dataframe.pkl')
r2 = pd.read_pickle('/Users/xies/Box/Mouse/Skin/W-R2/tracked_cells/dataframe.pkl')
r5 = pd.read_pickle('/Users/xies/Box/Mouse/Skin/W-R5/tracked_cells/dataframe.pkl')
df = pd.concat((r1,r2,r5))

# Load growth curves from pickle
with open('/Users/xies/Box/Mouse/Skin/W-R1/tracked_cells/collated_manual.pkl','rb') as f:
    c1 = pkl.load(f)
with open('/Users/xies/Box/Mouse/Skin/W-R2/tracked_cells/collated_manual.pkl','rb') as f:
    c2 = pkl.load(f)
with open('/Users/xies/Box/Mouse/Skin/W-R5/tracked_cells/collated_manual.pkl','rb') as f:
    c5 = pkl.load(f)
collated = c2

#######################################
# Look at error/noise via repeat measurements

repeat_dir = '/Users/xies/Box/Mouse/Skin/W-R2/repeat_tracked_cells/'

# Grab single-frame data into a dataframe
repeat_df = pd.DataFrame()
frames = []
cIDs = []
vols = []
fucci = []

filelist = glob(path.join(repeat_dir,'*/*.txt'))
for fullname in filelist:
    subdir,f = path.split(fullname)
    # Skip the log.txt or skipped.txt file
    if f == 'log.txt' or f == 'skipped.txt' or f == 'g1_frame.txt':
        continue
    if path.splitext(fullname)[1] == '.txt':
        print fullname
        # Grab the frame # from filename
        frame = f.split('.')[0]
        frame = np.int(frame[1:])
        frames.append(frame)
        
        # Grab cellID from subdir name
        cIDs.append( np.int(path.split(subdir)[1]) )
        
        # Add segmented area to get volume (um3)
        # Add total FUCCI signal to dataframe
        cell = pd.read_csv(fullname,delimiter='\t',index_col=0)
        vols.append(cell['Area'].sum())
repeat_df['Frame'] = frames
repeat_df['CellID'] = cIDs
repeat_df['Volume'] = vols


# Collate cell-centric list-of-dataslices
ucellIDs = np.unique( repeat_df['CellID'] )
Ncells = len(ucellIDs)
repeat = []
comparison = []

cIDs = [c.CellID[0] for c in collated]
for c in ucellIDs:
    this_cell = repeat_df[repeat_df['CellID'] == c].sort_values(by='Frame').copy()
    this_cell = this_cell.reset_index()
    repeat.append(this_cell)
    
    this_cell = collated[ np.where(cIDs == c)[0][0] ]
    this_cell = this_cell[this_cell['Daughter'] == 'None']
    comparison.append(this_cell)

repeat_lengths = [len(c) for c in repeat]
comparison_lengths = [len(c) for c in comparison]

repeat_volumes = np.concatenate( [c.Volume.values for c in repeat] )
comparison_volumes = np.concatenate( [c.Volume.values for c in comparison] )

# Rsquared
m,b,r,p,stderr = stats.linregress(comparison_volumes,repeat_volumes)
print 'R^2: ', r**2

plt.figure()
plt.scatter(repeat_volumes,comparison_volumes)
plt.plot([100,800],[100,800])
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('Original volume')
plt.ylabel('Repeat volume')

plt.figure()
plt.scatter(comparison_volumes, 100 * np.abs(repeat_volumes-comparison_volumes)/comparison_volumes)
plt.hlines(np.mean(100 * np.abs(repeat_volumes-comparison_volumes)/comparison_volumes),100,800,linestyles='dashed')
plt.plot([100,800],[100,800])
plt.xlabel('Original volume')
plt.ylabel('Absolute Error %')
plt.ylim([0,100])

print "Absolute error %:", np.mean(100 * np.abs(repeat_volumes-comparison_volumes)/comparison_volumes)

# Do the plotting nicely with dataframes
comparison_df = raw_df[np.in1d(raw_df['CellID'],cIDs)].copy()
comparison_df['CellID Frame'] = comparison_df['CellID'].astype(str) + '.'
comparison_df['CellID Frame'] = comparison_df['CellID Frame'] + comparison_df['Frame'].astype(str)
comparison_df.index = comparison_df['CellID Frame']
comparison_df['Repeat'] = 1

repeat_df['CellID Frame'] = repeat_df['CellID'].astype(str) + '.'
repeat_df['CellID Frame'] = repeat_df['CellID Frame'] + repeat_df['Frame'].astype(str)
repeat_df.index = repeat_df['CellID Frame']
repeat_df['Repeat'] = 2

repeat_df = repeat_df.join( comparison_df,lsuffix='_repeat',rsuffix='_original')

repeat_df['Vol diff'] = repeat_df['Volume_repeat'] - repeat_df['Volume_original']
repeat_df['Normed vol diff'] = abs(repeat_df['Vol diff'] / repeat_df['Volume_original']) * 100


sb.lmplot(data=repeat_df,x='Volume_original',y='Volume_repeat',hue='CellID_original',fit_reg=False)
plt.plot([100,800],[100,800])

sb.lmplot(data=repeat_df,x='Volume_original',y='Normed vol diff',hue='CellID_original',fit_reg=False)

plt.ylabel('Absolute Error %')
plt.ylim([0,100])


