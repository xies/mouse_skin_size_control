#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 16:05:31 2019

@author: xies
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pickle as pkl
from scipy import optimize
from scipy.interpolate import UnivariateSpline


###### Use spline fit to grab 

smoothing_factor = 1e5

for i,c in enumerate(collated):
    c['Region CellID'] = c['Region'] + c['CellID'].apply(str)
    collated[i]

yhat_spl = dict()
res_spl = []
n_knots = []
#counter = 0
# Fit Exponential & linear models to growth curves
for c in collated:
    if c.loc[0]['Phase'] != '?':
        c = c[c['Daughter'] == 'None']
        if len(c) > 3:
        #        t = np.arange(-g1sframe + 1,len(c) - g1sframe + 1) * 12 # In hours
            t = np.arange(len(c)) * 12
            v = c.Volume
        
            # B-spline
            spl = UnivariateSpline(t, v, k=3, s=smoothing_factor)
            yhat_spl[c.iloc[0]['Region CellID']] = spl(t)
            n_knots.append(len(spl.get_knots()))
            res_spl.append( (v-spl(t))/v )

all_res_spl = np.hstack(res_spl)

birth_vol = np.zeros(len(yhat_spl))
g1exit_vol = np.zeros(len(yhat_spl))
div_vol = np.zeros(len(yhat_spl))
div_vol_interp = np.zeros(len(yhat_spl))
ucellIDs = [ c.iloc[0]['Region CellID'] for c in collated ]

for i,cellID in enumerate(yhat_spl.keys()):
    
    yhat = yhat_spl[cellID]
    c = collated[ np.where(np.array([c == cellID for c in ucellIDs]))[0][0] ]
    birth_vol[i] = c.iloc[0].Volume
    g1exitframe = np.where(c['Phase'] == 'G1')[0][-1]
    g1exit_vol[i] = yhat[g1exitframe]
    div_vol[i] = c[c['Daughter'] == 'None'].iloc[-1]['Volume']
    div_vol[i] = c[c['Daughter'] == 'None'].iloc[-1]['Volume']
    div_vol_interp[i] = c[c['Daughter'] != 'None']['Volume'].sum()


##### Plotting 
plt.figure()
nbins = 5
g1_vol_bins = stats.mstats.mquantiles(g1exit_vol, np.arange(0,nbins+1,dtype=np.float)/nbins)

#S/G2 growth
# Final volume
sb.regplot(data = df, x= 'G1 volume',y='Division volume',fit_reg=True,ci=None)
plot_bin_means(df['G1 volume'],df['Division volume'],g1_vol_bins)
sb.regplot(df['G1 volume interpolated'],df['Division volume'],fit_reg=True,ci=None)
plot_bin_means(df['G1 volume interpolated'],df['Division volume'],g1_vol_bins)
plt.xlabel('G1 exit volume (original/interp) (um3)')
plt.ylabel('S/G2/M growth (original/interp) (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(('Original data','Spline-smoothed'))

plt.figure()
nbins = 5
birth_vol_bins = stats.mstats.mquantiles(birth_vol, np.arange(0,nbins+1,dtype=np.float)/nbins)

#G1 growth
# G1 volume
sb.regplot(df['Birth volume'],df['G1 volume'],fit_reg=True,ci=None)
plot_bin_means(df['Birth volume'],df['G1 volume'],birth_vol_bins)
sb.regplot(df['Birth volume'],df['G1 volume interpolated'],fit_reg=True,ci=None)
plot_bin_means(df['Birth volume'],df['G1 volume interpolated'],birth_vol_bins)
plt.xlabel('Birth volume (original) (um3)')
plt.ylabel('G1 exit volume (original/interp) (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(('Original data','Spline-smoothed'))

# Linear regression
x = df['Birth volume']
y = df['G1 volume interpolated']
I = ~(np.isnan(y) | np.isnan(x))
Pg1volinterp = np.polyfit(x[I],y[I],1)
print 'Slope of G1 volume: ', Pg1volinterp[0]
# Pearson
Rg1volinterp,P = stats.stats.pearsonr(x[I],y[I])
print 'Correlation of G1 volume: ', Rg1volinterp,P


# G1 growth
sb.regplot(df['Birth volume'],df['G1 grown'],fit_reg=True,ci=None)
#plot_bin_means(df['Birth volume'],df['G1 volume'],birth_vol_bins)
sb.regplot(df['Birth volume'],df['G1 volume interpolated']-df['Birth volume'],fit_reg=True,ci=None)
plot_bin_means(df['Birth volume'],df['G1 volume interpolated']-df['Birth volume'],birth_vol_bins)
plt.xlabel('Birth volume (original) (um3)')
plt.ylabel('G1 growth (original/interp) (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(('Original data','Spline-smoothed'))

# Linear regression
x = df['Birth volume']
y = df['G1 volume interpolated']-df['Birth volume']
I = ~(np.isnan(y) | np.isnan(x))
Pg1growthinterp = np.polyfit(x[I],y[I],1)
print 'Slope of G1 growth: ', Pg1growthinterp[0]
# Pearson
Rg1growthinterp,P = stats.stats.pearsonr(x[I],y[I])
print 'Correlation of G1 growth: ', Rg1growthinterp,P




