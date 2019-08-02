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

dx = 0.25

###### Use spline fit to grab 

smoothing_factor = 1e5

for i,c in enumerate(collated):
    c['Region CellID'] = c['Region'] + c['CellID'].apply(str)
    collated[i]

yhat_spl = dict()
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

sb.regplot(data = df, x= 'G1 volume',y='SG2 grown',fit_reg=False)
plot_bin_means(df['G1 volume'],df['SG2 grown'],g1_vol_bins)
sb.regplot(df['G1 volume interpolated'],df['Division volume'] - df['G1 volume interpolated'],fit_reg=False)
plot_bin_means(df['G1 volume interpolated'],df['Division volume'] - df['G1 volume interpolated'],g1_vol_bins)
plt.xlabel('G1 exit volume (original/interp) (um3)')
plt.ylabel('S/G2/M growth (original/interp) (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(('Original data','Spline-smoothed'))

plt.figure()
nbins = 5
birth_vol_bins = stats.mstats.mquantiles(birth_vol, np.arange(0,nbins+1,dtype=np.float)/nbins)

plt.scatter(df['Birth volume'],df['G1 grown'])
plot_bin_means(df['Birth volume'],df['G1 grown'],birth_vol_bins)
plt.scatter(df['Birth volume'],df['G1 volume interpolated'] - df['Birth volume'])
plot_bin_means(df['Birth volume'],df['G1 volume interpolated'] - df['Birth volume'],birth_vol_bins)

plt.xlabel('Birth volume (original) (um3)')
plt.ylabel('G1 growth (original/interp) (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(('Original data','Spline-smoothed'))






