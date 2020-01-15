#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:11:41 2019

@author: xies
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
g1_growth_bins = stats.mstats.mquantiles(df['G1 grown interpolated'],  np.arange(0,nbins+1,dtype=np.float)/nbins)
g2_growth_bins = stats.mstats.mquantiles(nonans(df['SG2 grown interpolated']), np.arange(0,nbins+1,dtype=np.float)/nbins)

# By by fix-width
#birth_vol_bins = np.linspace(df['Birth volume'].min(),df['Birth volume'].max() , nbins+1)
#g1_vol_bins = np.linspace(df['G1 volume'].min(),df['G1 volume'].max() , nbins+1)


## Find partial correlations
x = df['Birth volume']
y = df['G1 length']
p = np.polyfit(x,y,1)
residuals = np.polyval(p,x) - y
sb.regplot(residuals,df['SG2 length'])
plt.xlabel('Residual from linreg: birth volume v. G1 duration')
plt.ylabel('S/G2/M duration')

stats.stats.pearsonr(residuals,df['SG2 length'])
