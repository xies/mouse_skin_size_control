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

# USE load_regions.py to read data

# Construct histogram bins
nbins = 5
birth_vol_bins = stats.mstats.mquantiles(df['Birth volume'],  np.arange(0,nbins+1,dtype=np.float)/nbins)
g1_vol_bins = stats.mstats.mquantiles(df['G1 volume'], np.arange(0,nbins+1,dtype=np.float)/nbins)
#
################### Calculate growth rates ##################
##No smoothing
#for c in collated_filtered:
#    V = c.Volume
#    Tb = backward_difference(len(V)) # Generate Toeplitz matrix
#    dV = np.dot(Tb,V)
#    dV[0] = np.nan # first element is not meaningful
#    dV[c['Daughter'] != 'None'] = np.nan
#    c['Growth rate'] = dV / 12 #normalize to 12hrs
    
##smoothing
#for c in collated_filtered:
#    V = c.Volume
#    if len(V) > 3:
#        smdV = signal.savgol_filter(V,3,2,1)
##        Tc = central_difference(len(V)) # Generate Toeplitz matrix
##        dV = np.dot(Tc,V)
#        c['Sm Growth rate'] = smdV
#    else:
#        c['Sm Growth rate'] = np.nan
    
########## Plot histogram across growth curves ############

dfc = pd.concat(collated_filtered)
dfc = dfc[dfc['Phase'] != '?']
dfc.groupby('Phase')['Growth rate'].hist(stacked=True)
plt.xlabel('Growth rate (um3 / hr)')
plt.legend(('G1','SG2','M'))


########## Scatter plot across smoothed growth curves ############
x = dfc[dfc['Phase'] == 'G1']
g = sb.regplot(data=x,y = 'Growth rate (sm)',x = 'Volume (sm)',fit_reg=False,ci=None)
bins = stats.mstats.mquantiles(x['Volume'],np.array([0,1.,2.,3.,4.,5.,6.,7.])/7)
bins = np.linspace(x['Volume (sm)'].min(),x['Volume (sm)'].max(),8)
plot_bin_means(x['Volume'],x['Growth rate (sm)'],bins[:-2],color='b', error='std',style='fill')
x = dfc[dfc['Phase'] == 'SG2']
g = sb.regplot(data=x,y = 'Growth rate (sm)',x = 'Volume (sm)',fit_reg=False,ci=None)
bins = np.linspace(x['Volume'].min(),x['Volume (sm)'].max(),8)
plot_bin_means(x['Volume'],x['Growth rate (sm)'],bins[:-2],color='r', error='std',style='fill')

# Get fitting statistics
x = dfc[dfc['Phase'] != 'Daughter G1']['Volume (sm)']
x = dfc[dfc['Phase'] != 'Daughter G1']['Growth rate (sm)']
I = (~np.isnan(x)) & (~np.isnan(y))
slope, intercept, _,_,_ = stats.linregress(x[I],y[I])

########## Scatter plot across raw growth curves ############
x = dfc[dfc['Phase'] == 'G1']
g = sb.regplot(data=x,y = 'Growth rate',x = 'Volume',fit_reg=False,ci=None)
bins = stats.mstats.mquantiles(x['Volume'],np.array([0,1.,2.,3.,4.,5.,6.,7.])/7)
bins = np.linspace(x['Volume'].min(),x['Volume'].max(),8)
plot_bin_means(x['Volume'],x['Growth rate'],bins[:-2],color='b', error='std',style='fill')
x = dfc[dfc['Phase'] == 'SG2']
g = sb.regplot(data=x,y = 'Growth rate',x = 'Volume',fit_reg=False,ci=None)
bins = np.linspace(x['Volume'].min(),x['Volume'].max(),8)
plot_bin_means(x['Volume'],x['Growth rate'],bins[:-2],color='r', error='std',style='fill')


########## Plot geometric mean of growth curves as a function of size 

gmean_gr = np.array([stats.gmean(np.abs(nonans(c['Growth rate']))) for c in collated_filtered])
sb.regplot(df['Birth volume'],gmean_gr)
plt.xlabel('Birth volume')
plt.ylabel('Geometric mean of raw growth rates')



