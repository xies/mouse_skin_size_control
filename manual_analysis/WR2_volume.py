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

dirname = '/Users/xies/Box/Mouse/Skin/W-R2/tracked_cells/'

# Grab single-frame data into a dataframe
raw_df = pd.DataFrame()
frames = []
cIDs = []
vols = []
fucci = []
for subdir, dirs, files in os.walk(dirname):
    for f in files:
        fullname = os.path.join(subdir, f)
        # Skip the log.txt or skipped.txt file
        if f == 'log.txt' or f == 'skipped.txt' or f == 'g1_frame.txt':
            continue
        if os.path.splitext(fullname)[1] == '.txt':
            print fullname
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


# Collate cell-centric list-of-dataslices
ucellIDs = np.unique( raw_df['CellID'] )
Ncells = len(ucellIDs)
collated = []
for c in ucellIDs:
    this_cell = raw_df[raw_df['CellID'] == c].sort_values(by='Frame').copy()
    this_cell = this_cell.reset_index()
    collated.append(this_cell)

##### Export growth traces in CSV ######
pd.concat(collated).to_csv(path.join(dirname,'growth_curves.csv'),
                        index=False)

f = open(path.join(dirname,'collated_manual.pkl'),'w')
pkl.dump(collated,f)

# Load hand-annotated G1/S transition frame
g1transitions = pd.read_csv(
        path.join(dirname,'g1_frame.txt'))


# Collapse into single cell v. measurement DataFrame
Tcycle = np.zeros(Ncells)
Bsize = np.zeros(Ncells)
DivSize = np.zeros(Ncells)
G1duration = np.zeros(Ncells)
G1size = np.zeros(Ncells)
cIDs = np.zeros(Ncells)
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

df_nans = df
df = df[~np.isnan(df['G1 grown'])]

# Construct histogram bins
birth_vol_bins = stats.mstats.mquantiles(df['Birth volume'], [0, 1./6, 2./6, 3./6, 4./6, 6./6, 1])
g1_vol_bins = stats.mstats.mquantiles(df['G1 volume'], [0, 1./6, 2./6, 3./6, 4./6, 6./6, 1])

df['Region'] = 'M1R2'
r2 = df

#Pickle the dataframe
r2.to_pickle(path.join(dirname,'dataframe.pkl'))

#Load from pickle
r2 = pd.read_pickle(path.join(dirname,'dataframe.pkl'))

################## Plotting ##################

## Amt grown
plt.figure()
sb.set_style("darkgrid")

#plt.subplot(2,1,1)
sb.regplot(data=df,x='Birth volume',y='G1 grown',fit_reg=False)
plot_bin_means(df['Birth volume'],df['G1 grown'],birth_vol_bins)

#plt.subplot(2,1,2)
sb.regplot(data=df,x='G1 volume',y='SG2 grown',fit_reg=False)
plot_bin_means(df['G1 volume'],df['SG2 grown'],g1_vol_bins)
plt.xlabel('Volume at phase start (um3)')
plt.ylabel('Volume grown during phase (um3)')
plt.legend(['G1','SG2'])

plt.figure()
sb.regplot(data=df,x='Birth volume',y='G1 volume',fit_reg=False)
plot_bin_means(df['Birth volume'],df['G1 volume'],birth_vol_bins)
sb.regplot(data=df,x='G1 volume',y='Division volume',fit_reg=False)
plot_bin_means(df['G1 volume'],df['Division volume'],g1_vol_bins)
plt.xlabel('Volume at phase start (um3)')
plt.ylabel('Volume at phase end (um3)')
plt.legend(['G1','SG2'])


##
plt.figure()
sb.regplot(data=df,x='Birth volume',y='G1 volume',fit_reg=False)
plot_bin_means(df['Birth volume'],df['G1 volume'],birth_vol_bins)

## Adder?
plt.figure()
plt.subplot(2,1,1)
sb.regplot(data=df,x='Birth volume',y='Total growth',fit_reg=False)
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
    ax1.plot(x,v/v[0],color='b') # growth curve
    ax1.plot(x[-1], v[-1]/v[0],'ko',alpha=0.5) # end of growth
ax1.hlines(1,0,12,linestyles='dashed')
ax1.hlines(2,0,12,linestyles='dashed')
plt.ylabel('Fold grown since birth')

out = ax2.hist(df['Fold grown'], orientation="horizontal");

which_bin = np.digitize(df['Fold grown'],out[1])

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.plot(collated[i]['G1'])
    
    

#######################################
# Grab the automatic trancked data and look at how they relate

with open(path.join(dirname,'collated.pkl'),'rb') as f:
    auto_tracked = pkl.load(f)
autoIDs = np.array([c.CellID.iloc[0] for c in auto_tracked])
auto = []
for i in range(Ncells):
    ind = np.where(autoIDs == collated[i].CellID[0])[0][0]
    auto.append(auto_tracked[ind])

indices = np.arange(15,20,1)
for i in range(5):
    idx = indices[i]
    plt.subplot(2,5,i+1)
    plt.plot(auto[idx].Timeframe,auto[idx].ActinSegmentationArea,marker='o')
    plt.title(''.join( ('Cell #', str(auto[idx].CellID.iloc[0])) ))
    plt.ylabel('Cross section area')
    plt.subplot(2,5,i+6)
    plt.plot(collated[idx].Frame,collated[idx].Volume,color='g',marker='o')
    plt.ylabel('Volume')
    

Ncells = len(ucellIDs)
Aauto = np.empty((Ncells,10)) * np.nan
Vmanual = np.empty((Ncells,10)) * np.nan
for i in range(Ncells):
    a = auto[i].ActinSegmentationArea
    Aauto[i,0:len(a)] = a
    v = collated[i].Volume
    Vmanual[i,0:len(v)] = v
    if len(a) == len(v): #Sometimes automated tracker is mis-tracked
        print i
        plt.scatter(a,v,color='b')
        plt.xlabel('Cross sectional area')
        plt.ylabel('Volume')
        

Aautodiff = np.ndarray.flatten(np.diff(Aauto))
ax1 = plt.hist( nonans( Aautodiff/np.nanmean(Aauto) ), bins=25, histtype='step')
plt.vlines( np.nanmean( Aautodiff/np.nanmean(Aauto) ), 0,300, linestyles='dashed')

Vmanualdiff = np.ndarray.flatten(np.diff(Vmanual)) 
plt.hist( nonans( Vmanualdiff/np.nanmean(Vmanual) ),bins = 25, histtype='step')
plt.vlines( np.nanmean( Vmanualdiff/np.nanmean(Vmanual) ), 0,300, linestyles='dashed',color='r') 

plt.legend('Area','Volume')
plt.xlabel('dA/dt / <A> or dV/dt / <V>')
plt.ylabel('Count')


#######################################
# Look at error/noise via repeat measurements

repeat_dir = '/Users/xies/Box/Mouse/Skin/W-R2/repeat_tracked_cells/'

# Grab single-frame data into a dataframe
repeat_df = pd.DataFrame()
frames = []
cIDs = []
vols = []
fucci = []
for subdir, dirs, files in os.walk(repeat_dir):
    for f in files:
        fullname = os.path.join(subdir, f)
        # Skip the log.txt or skipped.txt file
        if f == 'log.txt' or f == 'skipped.txt' or f == 'g1_frame.txt':
            continue
        if os.path.splitext(fullname)[1] == '.txt':
            print fullname
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
    
    this_cell = collated[ np.where(cIDs == c)[0]]
    comparison.append(this_cell)

repeat_lengths = [len(c) for c in repeat]
comparison_lengths = [len(c) for c in comparison]

repeat_volumes = np.concatenate( [c.Volume.values for c in repeat] )
comparison_volumes = np.concatenate( [c.Volume.values for c in comparison] )

plt.scatter(repeat_volumes,comparison_volumes)
plt.plot([100,800],[100,800])
plt.xlabel('Original volume')
plt.ylabel('Repeat volume')


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
plt.hlines(repeat_df['Normed vol diff'].mean(),100,800,linestyles='dashed')
plt.ylabel('Absolute Error %')



