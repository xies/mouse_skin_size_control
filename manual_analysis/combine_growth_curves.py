#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:30:05 2019

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats, signal
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
collated = c1 + c2 + c5

####################################

# construct growth curves aligned at g1/s
g1lengths = pd.DataFrame([len(c[c['Phase'] == 'G1']) for c in collated])
sg2lengths = pd.DataFrame([len(c[c['Phase'] == 'SG2']) for c in collated])
phase_ambiguous = np.array([c['Phase'].iloc[0] == '?' for c in collated])
g1lengths[phase_ambiguous] = '?'
sg2lengths[phase_ambiguous] = '?'
Ncells = (g1lengths != '?').sum().values[0]

max_g1 = int(g1lengths[g1lengths != '?'].max().values[0])
max_sg2 = int(sg2lengths[g1lengths != '?'].max().values[0])

#initialize array with NaN
g1exit_aligned = np.empty((Ncells,max_g1+max_sg2)) * np.nan
collated_filtered = [c for c in collated if c['Phase'].iloc[0] != '?']

g1_aligned_frame = max_g1
for i,c in enumerate(collated_filtered):
    this_phase = c['Phase']
    this_g1_vol = c[this_phase == 'G1']['Volume']
    this_sg2_vol = c[this_phase == 'SG2']['Volume']
    g1exit_aligned[i,g1_aligned_frame-len(this_g1_vol):g1_aligned_frame] = this_g1_vol
    g1exit_aligned[i,g1_aligned_frame:g1_aligned_frame+len(this_sg2_vol)] = this_sg2_vol

####################################

# Plot G1-aligned growth curves
plt.pcolor(g1exit_aligned) # Heatmap
plt

# Plot growth curve(s): Region 1
fig=plt.figure()
ax1 = plt.subplot(121)
plt.xlabel('Time since birth (hr)')
plt.ylabel('Volume (um3)')
ax2 = plt.subplot(122)
curve_colors = {'M1R1':'b','M1R2':'r','M2R5':'g'}
for c in c1+c2+c5:
    v = np.array(c['Volume'],dtype=np.float)
    x = np.array(xrange(len(v))) * 12
    ax1.plot(x,v,alpha=0.2,color=curve_colors[c.iloc[0].Region]) # growth curve
#    ax1.plot(x[-1], v[-1]/v[0],'ko',alpha=0.5) # end of growth
out = ax2.hist(df['Fold grown'], orientation="vertical")
plt.xlabel('Fold grown from birth to division')

# Plot growth curve(s): Region 1
ax1 = plt.subplot(223)
plt.xlabel('Time since birth (hr)')
plt.ylabel('Volume (um3)')
for c in c2:
    v = np.array(c['Volume'],dtype=np.float)
    x = np.array(xrange(len(v))) * 12
    ax1.plot(x,v ,color='r') # growth curve
#    ax1.plot(x[-1], v[-1]/v[0],'ko',alpha=0.5) # end of growth
ax2 = plt.subplot(224)
out = ax2.hist(r2['Fold grown'], orientation="vertical",alpha=0.5)
