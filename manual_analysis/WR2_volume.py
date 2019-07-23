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
import os.path as path
import pickle as pkl
import re
from glob import glob
from scipy import stats

dirname = '/Users/xies/Box/Mouse/Skin/W-R2/tracked_cells/'
# Grab single-frame data into a dataframe
raw_df = pd.DataFrame()
frames = []
cIDs = []
vols = []
fucci = []
daughter = []
nuclei = []

dx = 0.25 # um per px

filelist = glob(path.join(dirname,'*/*.txt'))
for fullname in filelist:
    subdir,f = path.split(fullname)
    # Skip the log.txt or skipped.txt file
    if f == 'log.txt' or f == 'skipped.txt' or f == 'g1_frame.txt' or f == 'mitosis_in_frame.txt':
        continue
    fn, extension = path.splitext(f)
    if extension == '.txt':
        fn, channel = path.splitext(fn)
#            print fullname
        # Measure everything on FUCCI channel first
        if channel == '.fucci':
            subdir = path.split(subdir)[1]
            # Grab cellID from subdir name
            cIDs.append( np.int(path.split(subdir)[1]) )
            
            # Add segmented area to get volume (um3)
            # Add total FUCCI signal to dataframe
            cell = pd.read_csv(fullname,delimiter='\t',index_col=0)
            vols.append(cell['Area'].sum())
            fucci.append(cell['Mean'].mean())
            
            # Check if main lineage or daughter cells
            framename = path.split(fn)[1]
            match = re.search('(\D)$',framename)
            # Main cell linage
            if match == None:
                # Grab the frame # from filename
                frame = f.split('.')[0]
                frame = np.int(frame[1:])
                frames.append(frame)
                daughter.append('None')
            else:
                daughter_name = match.group(0)
                frame = f.split('.')[0]
                frame = np.int(frame[1:-1])
                frames.append(frame)
                daughter.append(daughter_name)
        elif channel == '.h2b':
            cell = pd.read_csv(fullname,delimiter='\t',index_col=0)
            nuclei.append(cell['IntDen'].sum().astype(np.float) * dx**2)
            
            
raw_df['Frame'] = frames
raw_df['CellID'] = cIDs
raw_df['Volume'] = vols
raw_df['Nucleus'] = nuclei
raw_df['G1'] = fucci
raw_df['Daughter'] = daughter

# Load hand-annotated G1/S transition frame
g1transitions = pd.read_csv(path.join(dirname,'g1_frame.txt'),',')

# Collate cell-centric list-of-dataslices
ucellIDs = np.unique( raw_df['CellID'] )
Ncells = len(ucellIDs)
collated = []
for c in ucellIDs:
    this_cell = raw_df[raw_df['CellID'] == c].sort_values(by='Frame').copy()
    this_cell['Region'] = 'M1R2'
    this_cell = this_cell.reset_index()
    # Annotate cell cycle of parent cell
    transition_frame = g1transitions[g1transitions.CellID == this_cell.CellID[0]].iloc[0].Frame
    if transition_frame == '?':
        this_cell['Phase'] = '?'
    else:
        this_cell['Phase'] = 'SG2'
        iloc = np.where(this_cell.Frame == np.int(transition_frame))[0][0]
        this_cell.loc[0:iloc,'Phase'] = 'G1'
    # Annotate cell cycle of daughter cell
    this_cell.loc[this_cell['Daughter'] != 'None','Phase'] = 'Daughter G1'
    collated.append(this_cell)
    
##### Export growth traces in CSV ######
pd.concat(collated).to_csv(path.join(dirname,'growth_curves.csv'),
                        index=False)

f = open(path.join(dirname,'collated_manual.pkl'),'w')
pkl.dump(collated,f)

# Load mitosis frame
mitosis_in_frame = pd.read_csv(path.join(dirname,'mitosis_in_frame.txt'),',')

# Collapse into single cell v. measurement DataFrame
Tcycle = np.zeros(Ncells)
Bsize = np.zeros(Ncells)
Bframe = np.zeros(Ncells)
DivSize = np.zeros(Ncells)
G1duration = np.zeros(Ncells)
G1size = np.zeros(Ncells)
cIDs = np.zeros(Ncells)
daughterSizes = np.zeros((2,Ncells))
for i,c in enumerate(collated):
    # Break out the daughter cells
    d = c[c['Daughter'] != 'None']
    c = c[c['Daughter'] == 'None']
    
    cIDs[i] = c['CellID'].iloc[0]
    Bsize[i] = c['Volume'].iloc[0]
    Bframe[i] = c['Frame'].iloc[0]
    DivSize[i] = c['Volume'][len(c)-1]
    Tcycle[i] = len(c) * 12
    # Find manual G1 annotation
    thisg1frame = g1transitions[g1transitions['CellID'] == c['CellID'].iloc[0]]['Frame'].values[0]
    if thisg1frame == '?':
        G1duration[i] = np.nan
        G1size[i] = np.nan
    else:
        thisg1frame = np.int(thisg1frame)
        G1duration[i] = (thisg1frame - c.iloc[0]['Frame'] + 1) * 12
        G1size[i] = c[c['Frame'] == thisg1frame]['Volume']
    # Annotate daughter cell data
    if len(d) > 0:
        daughterSizes[:,i] = d['Volume']
    else:
        daughterSizes[:,i] = np.nan
        
# Construct dataframe with primary data
df = pd.DataFrame()
df['CellID'] = cIDs
df['Birth frame'] = Bframe
df['Cycle length'] = Tcycle
df['G1 length'] = G1duration
df['G1 volume'] = G1size
df['Birth volume'] = Bsize
df['Division volume'] = DivSize
df['Daughter a volume'] = daughterSizes[0,:]
df['Daughter b volume'] = daughterSizes[1,:]

# Derive data
df['Daughter total volume'] = df['Daughter a volume'] + df['Daughter b volume']
df['Daughter ratio'] = np.vstack((df['Daughter a volume'], df['Daughter b volume'])).min(axis=0) / \
                        np.vstack((df['Daughter a volume'], df['Daughter b volume'])).max(axis=0)
df['Division volume interpolated'] = (df['Daughter total volume'] + df['Division volume'])/2
df['Total growth'] = df['Division volume'] - df['Birth volume']
df['SG2 length'] = df['Cycle length'] - df['G1 length']
df['G1 grown'] = df['G1 volume'] - df['Birth volume']
df['SG2 grown'] = df['Total growth'] - df['G1 grown']
df['Fold grown'] = df['Division volume'] / df['Birth volume']
df['Total growth interpolated'] = df['Division volume interpolated'] - df['Birth volume']


# Put in the mitosis annotation
df['Mitosis'] = np.in1d(df.CellID,mitosis_in_frame)
    
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
dirname = '/Users/xies/Box/Mouse/Skin/W-R2/'

with open(path.join(dirname,'collated.pkl'),'rb') as f:
    auto_tracked = pkl.load(f)
autoIDs = np.array([c.CellID.iloc[0] for c in auto_tracked])
auto = []
for i in range(Ncells):
    ind = np.where(autoIDs == collated[i].CellID.iloc[0])[0][0]
    auto.append(auto_tracked[ind])

indices = np.random.randint(len(auto),size=5)
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

