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
import statsmodels.api as sm

#### Uses interpolated cell volumes

################## Plotting ##################

# Construct histogram bins
# By by cohort
nbins = 5
birth_vol_bins = stats.mstats.mquantiles(df['Birth volume'],  np.arange(0,nbins+1,dtype=np.float)/nbins)
g1_vol_bins = stats.mstats.mquantiles(nonans(df['G1 volume interpolated']), np.arange(0,nbins+1,dtype=np.float)/nbins)

# By by fix-width
#birth_vol_bins = np.linspace(df['Birth volume'].min(),df['Birth volume'].max() , nbins+1)
#g1_vol_bins = np.linspace(df['G1 volume'].min(),df['G1 volume'].max() , nbins+1)


## Size control correlations
sb.set_style("darkgrid")

## Birth size G1 growth
sb.regplot(data=df,x='Birth volume',y='G1 grown',fit_reg=True,ci=None)
sb.regplot(data=df,x='Birth volume',y='G1 grown interpolated',fit_reg=True,ci=None)
plot_bin_means(df['Birth volume'],df['G1 grown interpolated'],birth_vol_bins)
plt.xlabel('Birth volume (um3)')
plt.ylabel('Amount grown in G1 (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([200,550])

## G1 exit volume SG2 growth
sb.regplot(data=df,x='G1 volume',y='SG2 grown',fit_reg=True,ci=None)
sb.regplot(data=df,x='G1 volume',y='SG2 grown interpolated',fit_reg=True,ci=None)
plot_bin_means(df['G1 volume'],df['SG2 grown interpolated'],g1_vol_bins)
plt.xlabel('G1 exit volume (um3)')
plt.ylabel('Amount grown in S/G2 (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([250,700])

## Birth size v G1 exit volume
sb.regplot(data=df,x='Birth volume',y='G1 volume',fit_reg=True,ci=None)
sb.regplot(data=df,x='Birth volume',y='G1 volume interpolated',fit_reg=True,ci=None)
plot_bin_means(df['Birth volume'],df['G1 volume interpolated'],birth_vol_bins)
plt.xlabel('Birth volume (um3)')
plt.ylabel('G1 exit volume (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([200,550])

## G1 exit volume v division volume
sb.regplot(data=df,x='G1 volume',y='Division volume',fit_reg=True,ci=None)
sb.regplot(data=df,x='G1 volume interpolated',y='Division volume interpolated',fit_reg=True,ci=None)
plot_bin_means(df['G1 volume interpolated'],df['Division volume interpolated'],g1_vol_bins)
plt.xlabel('G1 exit volume (um3)')
plt.ylabel('Division volume (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([250,700])

## Birth volume v total growth
sb.regplot(data=df,x='Birth volume',y='Total growth',fit_reg=True,ci=None)
sb.regplot(data=df,x='Birth volume',y='Total growth interpolated',fit_reg=True,ci=None)
plot_bin_means(df['Birth volume'],df['Total growth interpolated'],birth_vol_bins)
plt.xlabel('Birth volume (um3)')
plt.ylabel('Total growth (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([200,550])

## Final volume
sb.regplot(data=df,x='Birth volume',y='Division volume',fit_reg=True,ci=None)
sb.regplot(data=df,x='Birth volume',y='Division volume interpolated',fit_reg=True,ci=None)
plot_bin_means(df['Birth volume'],df['Division volume interpolated'],birth_vol_bins)
plt.xlabel('Birth volume (um3)')
plt.ylabel('Division volume (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([200,550])


## Phase length
sb.lmplot(data=df,x='Birth volume',y='G1 length',y_jitter=True,fit_reg=False,ci=None)
plot_bin_means(df['Birth volume'],df['G1 length'],birth_vol_bins)
plt.ylabel('G1 duration (hr)')
plt.xlabel('Volume at birth (um^3)')

sb.lmplot(data=df,x='G1 volume interpolated',y='SG2 length',y_jitter=True,fit_reg=False,ci=None)
plot_bin_means(df['G1 volume interpolated'],df['SG2 length'],g1_vol_bins)
plt.ylabel('S/G2/M duration (hr)')
plt.xlabel('Volume at S phase entry (um^3)')


################################################
# Correlations / linear regression slopes

# G1 growth
x = df['Birth volume']
x_const = sm.add_constant(x)
y = df['G1 grown interpolated']
I = ~(np.isnan(y) | np.isnan(x))
M = sm.OLS(y,x_const,missing='drop').fit()
print 'Slope of G1 grown: ', M.params[1], ' +- ', M.conf_int(0.05).values[1,1] - M.params[1]
# Pearson
R,P = stats.stats.pearsonr(x[I],y[I])
print 'Correlation of G1 grown: ', R,P

# G1 exit volume
x = df['Birth volume']
x_const = sm.add_constant(x)
y = df['G1 volume interpolated']
I = ~(np.isnan(y) | np.isnan(x))
M = sm.OLS(y,x_const,missing='drop').fit()
print 'Slope of G1 volume: ', M.params[1], ' +- ', M.conf_int(0.05).values[1,1] - M.params[1]
# Pearson
R,P = stats.stats.pearsonr(x[I],y[I])
print 'Correlation of G1 volume: ', R,P

 
# SG2 growth
x = df['G1 volume interpolated']
x_const = sm.add_constant(x)
y = df['SG2 grown interpolated']
I = ~(np.isnan(y) | np.isnan(x))
M = sm.OLS(y,x_const,missing='drop').fit()
print 'Slope of SG2 grown: ', M.params[1], ' +- ', M.conf_int(0.05).values[1,1] - M.params[1]
# Pearson
R,P = stats.stats.pearsonr(x[I],y[I])
print 'Correlation of SG2 grown: ', R,P

# SG2 exit volume
x = df['G1 volume interpolated']
x_const = sm.add_constant(x)
y = df['Division volume interpolated']
I = ~(np.isnan(y) | np.isnan(x))
M = sm.OLS(y,x_const,missing='drop').fit()
print 'Slope of SG2 volume: ', M.params[1], ' +- ', M.conf_int(0.05).values[1,1] - M.params[1]
# Pearson
R,P = stats.stats.pearsonr(x[I],y[I])
print 'Correlation of SG2 volume: ', R,P



# Total growth
x = df['Birth volume']
x_const = sm.add_constant(x)
y = df['Total growth interpolated']
I = ~(np.isnan(y) | np.isnan(x))
M = sm.OLS(y,x_const,missing='drop').fit()
print 'Slope of total grown: ', M.params[1], ' +- ', M.conf_int(0.05).values[1,1] - M.params[1]
# Pearson
R,P = stats.stats.pearsonr(x[I],y[I])
print 'Correlation of total grown: ', R,P

# SG2 exit volume
x = df['Birth volume']
x_const = sm.add_constant(x)
y = df['Division volume interpolated']
I = ~(np.isnan(y) | np.isnan(x))
M = sm.OLS(y,x_const,missing='drop').fit()
print 'Slope of division volume: ', M.params[1], ' +- ', M.conf_int(0.05).values[1,1] - M.params[1]
# Pearson
R,P = stats.stats.pearsonr(x[I],y[I])
print 'Correlation of division volume: ', R,P




