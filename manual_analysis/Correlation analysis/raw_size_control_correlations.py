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

################## Plotting ##################

# Construct histogram bins
# By by cohort
nbins = 5
birth_vol_bins = stats.mstats.mquantiles(df['Birth volume'],  np.arange(0,nbins+1,dtype=np.float)/nbins)
g1_vol_bins = stats.mstats.mquantiles(df['G1 volume'], np.arange(0,nbins+1,dtype=np.float)/nbins)

# By by fix-width
#birth_vol_bins = np.linspace(df['Birth volume'].min(),df['Birth volume'].max() , nbins+1)
#g1_vol_bins = np.linspace(df['G1 volume'].min(),df['G1 volume'].max() , nbins+1)


## Size control correlations
sb.set_style("darkgrid")

## G1 growth
sb.lmplot(data=df,x='Birth volume',y='G1 grown',fit_reg=False,ci=None,hue='Region')
plot_bin_means(df['Birth volume'],df['G1 grown'],birth_vol_bins)
plt.xlabel('Birth volume (um3)')
plt.ylabel('Amount grown in G1 (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([200,550])

## SG2 growth
sb.lmplot(data=df,x='G1 volume',y='SG2 grown',fit_reg=True,ci=None,hue='Region')
plot_bin_means(df['G1 volume'],df['SG2 grown'],g1_vol_bins)
plt.xlabel('G1 exit volume (um3)')
plt.ylabel('Amount grown in S/G2 (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([250,700])

## G1 volume
sb.lmplot(data=df,x='Birth volume',y='G1 volume',fit_reg=False,hue='Region')
plot_bin_means(df['Birth volume'],df['G1 volume'],birth_vol_bins)
plt.xlabel('Birth volume (um3)')
plt.ylabel('G1 exit volume (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([200,550])

## SG2 volume
sb.lmplot(data=df,x='G1 volume',y='Division volume',fit_reg=True,ci=None,hue='Region')
plot_bin_means(df['G1 volume'],df['Division volume'],g1_vol_bins)
plt.xlabel('G1 exit volume (um3)')
plt.ylabel('Division volume (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([250,700])

## Overall growth
sb.lmplot(data=df,x='Birth volume',y='Total growth',fit_reg=False,ci=None,hue='Region')
plot_bin_means(df['Birth volume'],df['Total growth'],birth_vol_bins)
plt.xlabel('Birth volume (um3)')
plt.ylabel('Total growth (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([200,550])

## Final volume
sb.lmplot(data=df,x='Birth volume',y='Division volume',fit_reg=False,ci=None,hue='Region')
plot_bin_means(df['Birth volume'],df['Division volume'],birth_vol_bins)
plt.xlabel('Birth volume (um3)')
plt.ylabel('Division volume (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([200,550])


## Phase length
sb.lmplot(data=df,x='Birth volume',y='G1 length',y_jitter=True,fit_reg=False,ci=None,hue='Region')
plot_bin_means(df['Birth volume'],df['G1 length'],birth_vol_bins)
plt.ylabel('G1 duration (hr)')
plt.xlabel('Volume at birth (um^3)')

sb.lmplot(data=df,x='G1 volume',y='SG2 length',y_jitter=True,fit_reg=False,ci=None,hue='Region')
plot_bin_means(df['G1 volume'],df['SG2 length'],g1_vol_bins)
plt.ylabel('S/G2/M duration (hr)')
plt.xlabel('Volume at S phase entry (um^3)')

################################################
# Correlations / linear regression slopes

# Pearson correlation
Rg1growth,P = stats.stats.pearsonr(df['Birth volume'],df['G1 grown'])
print 'Correlation of G1 growth: ', Rg1growth,P
Rsg2growth,P = stats.stats.pearsonr(df['G1 volume'],df['SG2 grown'])
print 'Correlation of S/G2 growth: ', Rsg2growth,P

Rg1vol,P = stats.stats.pearsonr(df['Birth volume'],df['G1 volume'])
print 'Correlation of G1 volume: ', Rg1vol,P
Rsg2Rg1vol,P = stats.stats.pearsonr(df['G1 volume'],df['Division volume'])
print 'Correlation of S/G2 volume: ', Rsg2Rg1vol,P

Rtotalgrowth,P = stats.stats.pearsonr(df['Birth volume'],df['Total growth'])
print 'Correlation of total growth: ', Rtotalgrowth, P
Rdivisionvol,p = stats.stats.pearsonr(df['Birth volume'],df['Division volume'])
print 'Correlation of division volume: ', Rdivisionvol,P

Rg1length,P = stats.stats.pearsonr(df['Birth volume'],df['G1 length'])
print 'Correlation of G1 length: ', Rg1length,P
Rsg2length,P = stats.stats.pearsonr(df['G1 volume'],df['SG2 length'])
print 'Correlation of S/G2 length: ', Rsg2length,P

# Linear regression
Pg1growth = np.polyfit(df['Birth volume'],df['G1 grown'],1)
Psg2growth = np.polyfit(df['G1 volume'],df['SG2 grown'],1)
print 'Slope of G1 growth: ', Pg1growth[0]
print 'Slope of S/G2 growth: ', Psg2growth[0]

# Linear regression
mbg1volume = np.polyfit(df['Birth volume'],df['G1 volume'],1)
print 'Slope of birth volume v G1 volume: ', Pbg1volume[0]
Pg2divvolume = np.polyfit(df['G1 volume'],df['Division volume'],1)
print 'Slope of G1 volume v division volume: ', Pg2divvolume[0]


Ptotalgrowth = np.polyfit(df['Birth volume'],df['Total growth'],1)
Pdivisionvol = np.polyfit(df['Birth volume'],df['Division volume'],1)
print 'Slope of total growth: ', Ptotalgrowth[0]
print 'Slope of division volume: ', Pdivisionvol[0]

################################


