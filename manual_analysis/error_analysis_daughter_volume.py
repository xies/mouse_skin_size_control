#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:24:43 2019

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pickle as pkl

#Load df from pickle
r1 = pd.read_pickle('/Users/xies/Box/Mouse/Skin/W-R1/tracked_cells/dataframe.pkl')
r2 = pd.read_pickle('/Users/xies/Box/Mouse/Skin/W-R2/tracked_cells/dataframe.pkl')
r5 = pd.read_pickle('/Users/xies/Box/Mouse/Skin/W-R5/tracked_cells/dataframe.pkl')
df = pd.concat((r1,r2,r5))

df = df[~df.Mitosis]
Ncells = len(df)


# Load growth curves from pickle
with open('/Users/xies/Box/Mouse/Skin/W-R1/tracked_cells/collated_manual.pkl','rb') as f:
    c1 = pkl.load(f)
with open('/Users/xies/Box/Mouse/Skin/W-R2/tracked_cells/collated_manual.pkl','rb') as f:
    c2 = pkl.load(f)
with open('/Users/xies/Box/Mouse/Skin/W-R5/tracked_cells/collated_manual.pkl','rb') as f:
    c5 = pkl.load(f)
collated = c1 + c2 + c5

ucellIDs = np.array([c.iloc[0].CellID for c in collated])

########################################################################

# Plot daughter division asymmetry
df_has_daughter = df[~np.isnan(df['Daughter a volume'])]
plt.hist(nonans(df_has_daughter['Daughter ratio']))
plt.xlabel('Daughter volume ratio')
plt.ylabel('Frequency')


# Plot daughter 'ghost' growth point
has_daughter = [c for c in collated if np.any(c.Daughter != 'None')]
Ncells = len(has_daughter)
GC_ = np.zeros((Ncells,13))
for i,c in enumerate(has_daughter):
    # Collate into heatmap

    mainC = c[c['Daughter'] == 'None']
    daughters = c[c['Daughter'] != 'None']
    t = (mainC.Frame - mainC.iloc[0].Frame)*12
    plt.plot(t,mainC.Volume,'b')
    plt.plot([t.iloc[-1], t.iloc[-1] + 6],
         [mainC.iloc[-1].Volume,daughters.Volume.sum()],
         marker='o',linestyle='dashed',color='r')
    plt.xlabel('Time since birth (hr)')
    plt.ylabel('Cell volume')
    
    GC_[i,0:len(mainC)] = mainC.Volume
    GC_[i,len(mainC):len(mainC) + 1] = daughters.Volume.sum()

########################################################################
    
# Align growth curves at G1/S
collated_filtered = [c for c in has_daughter if c.iloc[0].Phase != '?']

g1lengths = pd.DataFrame([len(c[c['Phase'] == 'G1']) for c in collated_filtered])
sg2lengths = pd.DataFrame([len(c[c['Phase'] == 'SG2']) for c in collated_filtered])
phase_ambiguous = np.array([c['Phase'].iloc[0] == '?' for c in collated_filtered])
g1lengths[phase_ambiguous] = '?'
sg2lengths[phase_ambiguous] = '?'
Ncells = (g1lengths != '?').sum().values[0]

max_g1 = int(g1lengths[g1lengths != '?'].max().values[0])
max_sg2 = int(sg2lengths[g1lengths != '?'].max().values[0])

#initialize array with NaN
g1exit_aligned = np.empty((Ncells,max_g1+max_sg2)) * np.nan
g1_aligned_frame = max_g1
g1_notaligned = np.empty((Ncells,10)) * np.nan

for i,c in enumerate(collated_filtered):
    this_phase = c['Phase']
    v = c[c.Daughter == 'None'].Volume
    daughters = c[c['Daughter'] != 'None']
    # Volume
    this_g1_vol = c[this_phase == 'G1']['Volume']
    this_sg2_vol = c[this_phase == 'SG2']['Volume']
    daughter_vol = c[this_phase == 'Daughter G1'].Volume.sum()
    g1exit_aligned[i,g1_aligned_frame-len(this_g1_vol):g1_aligned_frame] = this_g1_vol
    g1exit_aligned[i,g1_aligned_frame:g1_aligned_frame+len(this_sg2_vol)] = this_sg2_vol
    g1exit_aligned[i,g1_aligned_frame+len(this_sg2_vol):g1_aligned_frame+len(this_sg2_vol)+1] = daughter_vol

t = np.arange(-max_g1 + 1 ,max_sg2 + 1) * 12
t_birth = np.arange(10) * 12


# Heatmap 
X,Y = np.meshgrid(t,range(Ncells))
plt.pcolor(X,Y,g1exit_aligned)

########################################################################

# Construct histogram bins
nbins = 5
birth_vol_bins = stats.mstats.mquantiles(df['Birth volume'],  np.arange(0,nbins+1,dtype=np.float)/nbins)
g1_vol_bins = stats.mstats.mquantiles(df['G1 volume'], np.arange(0,nbins+1,dtype=np.float)/nbins)


## Amt grown

sb.lmplot(data=df,x='G1 volume',y='SG2 grown',fit_reg=False)
plot_bin_means(df['G1 volume'],df['SG2 grown'],g1_vol_bins)
plt.xlabel('G1 exit volume (um3)')
plt.ylabel('Amount grown in S/G2 (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([250,850])

sb.lmplot(data=df,x='G1 volume',y='Division volume interpolated',fit_reg=False)
plot_bin_means(df['G1 volume'],df['Division volume interpolated'],g1_vol_bins)
plt.xlabel('G1 exit volume (um3)')
plt.ylabel('Division volume (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([250,650])




