#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 02:02:05 2019

@author: mimi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
import os.path as path
import pickle as pkl
from scipy import stats

dirname = '/Users/xies/Box/Mouse/Skin/W-R5/tracked_cells/'

# Grab single-frame data into a dataframe
raw_df = pd.DataFrame()
frames = []
cIDs = []
vols = []
fucci = []
method = []
for subdir, dirs, files in os.walk(dirname):
    for f in files:
        fullname = os.path.join(subdir, f)
        # Skip the log.txt or skipped.txt file
        if f == 'log.txt' or f == 'skipped.txt' or f == 'g1_frame.txt' or f == 'mitosis_in_frame.txt':
            continue
        fn, extension = os.path.splitext(fullname)
        if os.path.splitext(fullname)[1] == '.txt':
            fn, channel = os.path.splitext(fn)
            print fullname
            # Measure everything on DAPI channel first
            if channel == '.fucci':
                # Grab the frame # from filename
                frame = f.split('.')[0]
                frame = np.int(frame[1:])
                frames.append(frame)
                
                # Grab cellID from subdir name
                cIDs.append( np.int(os.path.split(subdir)[1]) )
                
                # Add segmented area to get volume (um3)
                # Add total FUCCI signal to dataframe
                cell = pd.read_csv(fullname,delimiter='\t',index_col=0)
                vols.append(cell['Area'].sum())
                fucci.append(cell['Mean'].mean())

man_dirname = '/Users/xies/Box/Mouse/Skin/W-R5/manual_tracking/'
# Grab single-frame data into a dataframe
for subdir, dirs, files in os.walk(man_dirname):
    for f in files:
        fullname = os.path.join(subdir, f)
        # Skip the log.txt or skipped.txt file
        if f == 'log.txt' or f == 'skipped.txt' or f == 'g1_frame.txt' or f == 'mitosis_in_frame.txt':
            continue
        fn, extension = os.path.splitext(fullname)
        if os.path.splitext(fullname)[1] == '.txt':
            fn, channel = os.path.splitext(fn)
            print fullname
            # Measure everything on DAPI channel first
            if channel == '.fucci':
                # Grab the frame # from filename
                frame = f.split('.')[0]
                frame = np.int(frame[1:])
                frames.append(frame)
                
                # Grab cellID from subdir name
                cIDs.append( np.int(os.path.split(subdir)[1]) )
                
                # Add segmented area to get volume (um3)
                # Add total FUCCI signal to dataframe
                cell = pd.read_csv(fullname,delimiter='\t',index_col=0)
                vols.append(cell['Area'].sum())
                fucci.append(cell['Mean'].mean())  

raw_df['Frame'] = frames
raw_df['CellID'] = cIDs
raw_df['Volume'] = vols
raw_df['G1'] = fucci

# Load hand-annotated G1/S transition frame
g1transitions = pd.read_csv(path.join(dirname,'g1_frame.txt'),',')
g1transitions = pd.concat((g1transitions,pd.read_csv(path.join(man_dirname,'g1_frame.txt'),',')))

# Load mitosis frame
mitosis_in_frame = pd.read_csv(path.join(dirname,'mitosis_in_frame.txt'),',')

# Collate cell-centric list-of-dataslices
ucellIDs = np.unique( raw_df['CellID'] )
Ncells = len(ucellIDs)
collated = []
for c in ucellIDs:
    this_cell = raw_df[raw_df['CellID'] == c].sort_values(by='Frame').copy()
    this_cell['Region'] = 'M2R5'
    this_cell = this_cell.reset_index()
    # Annotate cell cycle
    transition_frame = g1transitions[g1transitions.CellID == this_cell.CellID[0]].iloc[0].Frame
    if transition_frame == '?':
        this_cell['Phase'] = '?'
    else:
        this_cell['Phase'] = 'SG2'
        iloc = np.where(this_cell.Frame == np.int(transition_frame))[0][0]
        this_cell.loc[0:iloc,'Phase'] = 'G1'
    collated.append(this_cell)

##### Export growth traces in CSV ######
pd.concat(collated).to_csv(path.join(dirname,'growth_curves.csv'),
                        index=False)

f = open(path.join(dirname,'collated_manual.pkl'),'w')
pkl.dump(collated,f)

# Collapse into single cell v. measurement DataFrame
Tcycle = np.zeros(Ncells)
Bsize = np.zeros(Ncells)
DivSize = np.zeros(Ncells)
G1duration = np.zeros(Ncells)
G1size = np.zeros(Ncells)
cIDs = np.zeros(Ncells)
method = np.zeros(Ncells)
for i,c in enumerate(collated):
    cIDs[i] = c['CellID'][0]
    Bsize[i] = c['Volume'][0]
    DivSize[i] = c['Volume'][len(c)-1]
    Tcycle[i] = len(c) * 12
    # Find manual G1 annotation
    thisg1frame = g1transitions[g1transitions['CellID'] == c['CellID'][0]]['Frame'].values[0]
    if thisg1frame == '?':
        G1duration[i] = np.nan
        G1size[i] = np.nan
    else:
        thisg1frame = np.int(thisg1frame)
        G1duration[i] = (thisg1frame - c.iloc[0]['Frame'] + 1) * 12
        G1size[i] = c[c['Frame'] == thisg1frame]['Volume']

# Construct dataframe with primary data
df = pd.DataFrame()
df['CellID'] = cIDs
df['Cycle length'] = Tcycle
df['G1 length'] = G1duration
df['G1 volume'] = G1size
df['Birth volume'] = Bsize
df['Division volume'] = DivSize

# Derive convenient data
df['Total growth'] = df['Division volume'] - df['Birth volume']
df['SG2 length'] = df['Cycle length'] - df['G1 length']
df['G1 grown'] = df['G1 volume'] - df['Birth volume']
df['SG2 grown'] = df['Total growth'] - df['G1 grown']
df['Fold grown'] = df['Division volume'] / df['Birth volume']
method = np.array(['Auto' for x in range(len(df))])
method[df['CellID'] < 100] = 'Manual'
df['Method'] = method

df_nans = df
df = df[~np.isnan(df['G1 grown'])]

# Put in the mitosis annotation
df['Mitosis'] = np.in1d(df.CellID,mitosis_in_frame)

# Construct histogram bins
nbins = 4
birth_vol_bins = stats.mstats.mquantiles(df['Birth volume'],  np.arange(0,nbins+1,dtype=np.float)/nbins)
g1_vol_bins = stats.mstats.mquantiles(df['G1 volume'], np.arange(0,nbins+1,dtype=np.float)/nbins)

df['Region'] = 'M2R5'
r5 = df

#Pickle the dataframe
r5.to_pickle(path.join(dirname,'dataframe.pkl'))

#Load from pickle
r5 = pd.read_pickle(path.join(dirname,'dataframe.pkl'))

################## Plotting ##################

df = r5[~r5.Mitosis]

## Amt grown
plt.figure()
sb.set_style("darkgrid")

#plt.subplot(2,1,1)
sb.lmplot(data=df,x='Birth volume',y='G1 grown',fit_reg=False,hue='Method')
plot_bin_means(df['Birth volume'],df['G1 grown'],birth_vol_bins)

#plt.subplot(2,1,2)
sb.lmplot(data=df,x='G1 volume',y='SG2 grown',fit_reg=False,hue='Method')
plot_bin_means(df['G1 volume'],df['SG2 grown'],g1_vol_bins)
plt.xlabel('Volume at phase start (um3)')
plt.ylabel('Volume grown during phase (um3)')
plt.legend(['G1','SG2'])

plt.figure()
sb.regplot(data=df,x='Birth volume',y='G1 volume',fit_reg=False)
plot_bin_means(df['Birth volume'],df['G1 volume'],birth_vol_bins)
sb.regplot(data=df,x='G1 volume',y='Division volume',fit_reg=False)
plt.xlabel('Volume at phase start (um3)')
plot_bin_means(df['G1 volume'],df['Division volume'],g1_vol_bins)
plt.ylabel('Volume at phase end (um3)')
plt.legend(['G1','SG2'])

plt.figure()
## Adder?
plt.subplot(2,1,1)
sb.lmplot(data=df,x='Birth volume',y='Total growth',fit_reg=False,hue='Method')
plot_bin_means(df['Birth volume'],df['Total growth'],birth_vol_bins)
plt.subplot(2,1,2)
sb.regplot(data=df,x='Birth volume',y='Division volume',fit_reg=False)
plot_bin_means(df['Birth volume'],df['Division volume'],birth_vol_bins)


## Phase length
plt.figure()
#plt.subplot(2,1,1)
sb.regplot(data=df,x='Birth volume',y='G1 length',y_jitter=True,fit_reg=False)
plot_bin_means(df['Birth volume'],df['G1 length'],birth_vol_bins)
#plt.subplot(2,1,2)
sb.regplot(data=df,x='G1 volume',y='SG2 length',y_jitter=False,fit_reg=False)
plot_bin_means(df['G1 volume'],df['SG2 length'],g1_vol_bins)
plt.legend(['G1','SG2'])
plt.ylabel('Phase duration (hr)')
plt.xlabel('Volume at phase start (um^3)')



# Plot growth curve(s)
fig=plt.figure()
ax1 = plt.subplot(121)
plt.xlabel('Time since birth (hr)')
ax2 = plt.subplot(122, sharey = ax1)
for i in range(Ncells):
    v = np.array(collated[i]['Volume'],dtype=np.float)
    x = np.array(xrange(len(v))) * 12
    ax1.plot(x,v/v[0] ,color='b') # growth curve
    ax1.plot(x[-1], v[-1]/v[0],'ko',alpha=0.5) # end of growth
ax1.hlines(1,0,12,linestyles='dashed')
ax1.hlines(2,0,12,linestyles='dashed')
plt.ylabel('Fold grown since birth')
out = ax2.hist(df['Fold grown'], orientation="horizontal");

which_bin = np.digitize(df['Fold grown'],out[1])

I = np.arange(50,60)
for (i,idx) in enumerate(I):
    plt.subplot(2,5,i+1)
    v = collated[idx]['Volume']
    plt.plot(v/v[0])
    
    
#######################################
