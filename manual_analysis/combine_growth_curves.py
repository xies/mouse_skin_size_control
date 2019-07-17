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
g1exit_h2b = np.empty((Ncells,max_g1+max_sg2)) * np.nan
g1exit_fucci = np.empty((Ncells,max_g1+max_sg2)) * np.nan
collated_filtered = [c for c in collated if c['Phase'].iloc[0] != '?']

g1_aligned_frame = max_g1
for i,c in enumerate(collated_filtered):
    this_phase = c['Phase']
    # Volume
    this_g1_vol = c[this_phase == 'G1']['Volume']
    this_sg2_vol = c[this_phase == 'SG2']['Volume']
    g1exit_aligned[i,g1_aligned_frame-len(this_g1_vol):g1_aligned_frame] = this_g1_vol
    g1exit_aligned[i,g1_aligned_frame:g1_aligned_frame+len(this_sg2_vol)] = this_sg2_vol
    # H2B
    this_g1_h2b = c[this_phase == 'G1']['H2B']
    this_sg2_h2b = c[this_phase == 'SG2']['H2B']
    g1exit_h2b[i,g1_aligned_frame-len(this_g1_vol):g1_aligned_frame] = this_g1_h2b
    g1exit_h2b[i,g1_aligned_frame:g1_aligned_frame+len(this_sg2_vol)] = this_sg2_h2b
    # FUCCI
    this_g1_fucci = c[this_phase == 'G1']['G1']
    this_sg2_fucci = c[this_phase == 'SG2']['G1']
    g1exit_fucci[i,g1_aligned_frame-len(this_g1_vol):g1_aligned_frame] = this_g1_fucci
    g1exit_fucci[i,g1_aligned_frame:g1_aligned_frame+len(this_sg2_vol)] = this_sg2_fucci

####################################

t = np.arange(-max_g1 + 1 ,max_sg2 + 1) * 12
# Plot G1-aligned growth curves
X,Y = np.meshgrid(t,np.arange(1,Ncells + 1))
plt.pcolor(X,Y,g1exit_aligned) # Heatmap ->need to control meshgrid
plt.xlabel('Time since G1 exit (hr)')
plt.xlabel('Individual cells')
plt.colorbar

plt.figure()
for i in xrange(Ncells):
    plt.plot(t,g1exit_aligned[i,:],color='b',alpha=0.2)
# plot mean/error as shade
Ncell_in_bin = (~np.isnan(g1exit_aligned)).sum(axis=0)
mean_curve = np.nanmean(g1exit_aligned,axis=0)
mean_curve[Ncell_in_bin < 10] = np.nan
std_curve = np.nanstd(g1exit_aligned,axis=0)
std_curve[Ncell_in_bin < 10] = np.nan
plt.plot(t, mean_curve, color='r')
plt.fill_between(t, mean_curve-std_curve, mean_curve+std_curve,
                 color='r',alpha=0.1)
plt.xlabel('Time since G1 exit (hr)')
plt.ylabel('Cell volume (um3)')

# Plot G1-aligned CV
plt.figure()
t = np.arange(-max_g1 + 1 ,max_sg2 + 1) * 12
# plot mean/error as shade
Ncell_in_bin = (~np.isnan(g1exit_aligned)).sum(axis=0)
cv = stats.variation(g1exit_aligned,axis=0,nan_policy='omit')
cv[Ncell_in_bin < 10] = np.nan
plt.plot(t,cv)


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



