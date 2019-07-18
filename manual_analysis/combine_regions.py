#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:33:22 2019

@author: mimi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats
import pickle as pkl

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
collated = c1+c2+c5

df = df[~df.Mitosis]

################## Plotting ##################

# Construct histogram bins
nbins = 5
birth_vol_bins = stats.mstats.mquantiles(df['Birth volume'],  np.arange(0,nbins+1,dtype=np.float)/nbins)
g1_vol_bins = stats.mstats.mquantiles(df['G1 volume'], np.arange(0,nbins+1,dtype=np.float)/nbins)


## Amt grown
sb.set_style("darkgrid")

sb.lmplot(data=df,x='Birth volume',y='G1 grown',fit_reg=False,hue='Region')
plot_bin_means(df['Birth volume'],df['G1 grown'],birth_vol_bins)
plt.xlabel('Birth volume (um3)')
plt.ylabel('Amount grown in G1 (um3)')
sb.lmplot(data=df,x='G1 volume',y='SG2 grown',fit_reg=False,hue='Region')
plot_bin_means(df['G1 volume'],df['SG2 grown'],g1_vol_bins)
plt.xlabel('G1 exit volume (um3)')
plt.ylabel('Amount grown in S/G2 (um3)')

sb.lmplot(data=df,x='Birth volume',y='G1 volume',fit_reg=False,hue='Region')
plot_bin_means(df['Birth volume'],df['G1 volume'],birth_vol_bins)
plt.xlabel('Birth volume (um3)')
plt.ylabel('G1 exit volume (um3)')
sb.lmplot(data=df,x='G1 volume',y='Division volume',fit_reg=False,hue='Region')
plot_bin_means(df['G1 volume'],df['Division volume'],g1_vol_bins)
plt.xlabel('G1 exit volume (um3)')
plt.ylabel('Division volume (um3)')

# Pearson correlation
Rg1growth = np.corrcoef(df['Birth volume'],df['G1 grown'])
Rsg2growth = np.corrcoef(df['G1 volume'],df['SG2 grown'])
print 'Correlation of G1 growth: ', Rg1growth[0,1]
print 'Correlation of S/G2 growth: ', Rsg2growth[0,1]

## Overall Adder?
sb.lmplot(data=df,x='Birth volume',y='Total growth',fit_reg=False,hue='Region')
plot_bin_means(df['Birth volume'],df['Total growth'],birth_vol_bins)
plt.xlabel('Birth volume (um3)')
plt.ylabel('Total growth (um3)')

sb.lmplot(data=df,x='Birth volume',y='Division volume',fit_reg=False,hue='Region')
plot_bin_means(df['Birth volume'],df['Division volume'],birth_vol_bins)
plt.xlabel('Birth volume (um3)')
plt.ylabel('Division volume (um3)')



## Phase length
sb.lmplot(data=df,x='Birth volume',y='G1 length',y_jitter=True,fit_reg=False,hue='Region')
plot_bin_means(df['Birth volume'],df['G1 length'],birth_vol_bins)
plt.ylabel('G1 duration (hr)')
plt.xlabel('Volume at birth (um^3)')

sb.lmplot(data=df,x='G1 volume',y='SG2 length',y_jitter=True,fit_reg=False,hue='Region')
plot_bin_means(df['G1 volume'],df['SG2 length'],g1_vol_bins)
plt.ylabel('S/G2/M duration (hr)')
plt.xlabel('Volume at S phase entry (um^3)')

# Pearson correlation
Rg1length = np.corrcoef(df['Birth volume'],df['G1 length'])
Rsg2length = np.corrcoef(df['G1 volume'],df['SG2 length'])
print 'Correlation of G1 length: ', Rg1length[0,1]
print 'Correlation of S/G2 length: ', Rsg2length[0,1]

# Plot CV/variation
# Fraction of growth in G1
plt.figure()
plt.hist(df['G1 grown']/df['Total growth'])
plt.xlabel('Fraction of growth occuring in G1')
plt.ylabel('Frequency')

# Calculate CVs
birthCV = stats.variation(df['Birth volume'])
g1CV = stats.variation(df['G1 volume'])
divisionCV = stats.variation(df['Division volume'])

# Calculate dispersion index
birthFano = np.var(df['Birth volume']) / np.mean(df['Birth volume'])
g1Fano = np.var(df['G1 volume']) / np.mean(df['G1 volume'])
divisionFano = np.var(df['Division volume']) / np.mean(df['Division volume'])

# Calculate skew
birthSkew = stats.skew(df['Birth volume'])
g1Skew = stats.skew(df['G1 volume'])
divisionSkew = stats.skew(df['Division volume'])

sb.catplot(data=df.melt(id_vars='CellID',value_vars=['Birth volume','G1 volume','Division volume'],
                        value_name='Volume'),
           x='variable',y='Volume')

plt.figure()
plt.plot([birthCV,g1CV,divisionCV])
plt.xticks(np.arange(3), ['Birth volume','G1 volume','Division volume'])
plt.ylabel('Coefficient of variation')

plt.figure()
plt.plot([birthSkew,g1Skew,divisionSkew])
plt.xticks(np.arange(3), ['Birth volume','G1 volume','Division volume'])
plt.ylabel('Distribution skew')

########## Truncate mouse 1 ############
t1 = [c for c in c1 if c.Frame.min() > 6]
subsetIDs = [np.unique(c.CellID)[0] for c in t1]
r1_trunc = r1[ np.in1d(r1.CellID,subsetIDs)]
t2 = [c for c in c2 if c.Frame.min() > 6]
subsetIDs = [np.unique(c.CellID)[0] for c in t2]
r2_trunc = r2[ np.in1d(r2.CellID,subsetIDs)]

df = pd.concat((r1_trunc,r2_trunc,r5))

