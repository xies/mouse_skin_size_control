#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 18:04:57 2019

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

# Construct histogram bins
nbins = 5
birth_vol_bins = stats.mstats.mquantiles(df['Birth volume'],  np.arange(0,nbins+1,dtype=np.float)/nbins)
g1_vol_bins = stats.mstats.mquantiles(df['G1 volume'], np.arange(0,nbins+1,dtype=np.float)/nbins)

################## Calculate growth rates ##################
#No smoothing
for c in collated:
    V = c.Volume
    Tb = backward_difference(len(V)) # Generate Toeplitz matrix
    dV = np.dot(Tb,V)
    dV[0] = np.nan # first element is not meaningful
    c['Growth rate'] = dV / 12 #normalize to 12hrs

#smoothing
for c in collated:
    V = c.Volume
    if len(V) > 3:
        smdV = signal.savgol_filter(V,3,2,1)
#        Tc = central_difference(len(V)) # Generate Toeplitz matrix
#        dV = np.dot(Tc,V)
        c['Sm Growth rate'] = smdV
    else:
        c['Sm Growth rate'] = np.nan
    
########## Plot histogram across growth curves ############
        
df = pd.concat(collated)
df = df[df['Phase'] != '?']
df.groupby('Phase')['Growth rate'].hist(stacked=True)
plt.xlabel('Growth rate (um3 / hr)')


x = df[df['Phase'] != 'Daughter G1']
g = sb.lmplot(data=df[df['Phase'] != 'Daughter G1'],y = 'Growth rate',x = 'Volume',hue='Phase',fit_reg=False)
#sb.regplot(data=df[df['Phase'] != 'Daughter G1'],y = 'Growth rate',x = 'Volume', scatter=False, ax=g.axes[0, 0])
bins = stats.mstats.mquantiles(x['Volume'],np.array([0,1.,2.,3.,4.,5.,6.,7.])/7)
plot_bin_means(x['Volume'],x['Growth rate'],bins,color='r', error='std')

