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

#Load from pickle
r1 = pd.read_pickle('/Users/mimi/Box Sync/Mouse/Skin/W-R1/tracked_cells/dataframe.pkl')
r2 = pd.read_pickle('/Users/mimi/Box Sync/Mouse/Skin/W-R2/tracked_cells/dataframe.pkl')

df = pd.concat((r1,r2))

# Construct histogram bins
birth_vol_bins = stats.mstats.mquantiles(df['Birth volume'], [0, 1./6, 2./6, 3./6, 4./6, 6./6, 1])
g1_vol_bins = stats.mstats.mquantiles(df['G1 volume'], [0, 1./6, 2./6, 3./6, 4./6, 6./6, 1])

################## Plotting ##################

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
plt.figure()
#plt.subplot(2,1,1)
sb.regplot(data=r1,x='Birth volume',y='G1 length',y_jitter=True,fit_reg=False)
plot_bin_means(df['Birth volume'],df['G1 length'],birth_vol_bins)
#plt.subplot(2,1,2)
sb.regplot(data=r1,x='G1 volume',y='SG2 length',y_jitter=True,fit_reg=False)
plot_bin_means(df['G1 volume'],df['SG2 length'],g1_vol_bins)
plt.ylabel('Phase duration (hr)')
plt.xlabel('Volume at phase start (um^3)')

#### Load pickled collated growth curves

f = open('/Users/mimi/Box Sync/Mouse/Skin/W-R1/tracked_cells/collated.pkl','r')
c1 = pkl.load(f)
f = open('/Users/mimi/Box Sync/Mouse/Skin/W-R2/tracked_cells/collated.pkl','r')
c2 = pkl.load(f)


# Plot growth curve(s): Region 1
fig=plt.figure()
ax1 = plt.subplot(221)
plt.xlabel('Time since birth (hr)')
plt.ylabel('Volume (um3)')
ax2 = plt.subplot(222)
for c in c1:
    v = np.array(c['Volume'],dtype=np.float)
    x = np.array(xrange(len(v))) * 12
    ax1.plot(x,v ,color='b', alpha=0.5) # growth curve
#    ax1.plot(x[-1], v[-1]/v[0],'ko',alpha=0.5) # end of growth
out = ax2.hist(r1['Fold grown'], orientation="vertical")

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

 

