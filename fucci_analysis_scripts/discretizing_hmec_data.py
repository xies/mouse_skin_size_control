#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:07:42 2019

@author: xies
"""

import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sb

filename = '/Users/mimi/Box Sync/HMECs/HMEC DFB tracked data/pbs_area.csv'
pbs = pd.read_csv(filename,names=range(59))

filename = '/Users/mimi/Box Sync/HMECs/HMEC DFB tracked data/palbo_size.csv'
palbo = pd.read_csv(filename,names = ['Tg1','Bsize'])

####### Analyze whole data sets for sizer v adder
Bsize = pbs.iloc[0]
Dsize = np.zeros(59)
for i in range(59):
    x = pbs[i]
    Dsize[i] = np.array(x[~np.isnan(x)])[-1]
amtGrown = Dsize - Bsize
plt.subplot(2,1,1)
plt.scatter(Bsize,amtGrown)
plt.ylabel('Amount grown')
plt.subplot(2,1,2)
plt.scatter(Bsize,Dsize)
plt.ylabel('Division size')
plt.xlabel('Birth size')

###### Need to decimate data by factor of 20 (with random phases)
dec_factor = 20.
Ttotal = len(pbs[0])
PBS = np.zeros((59, np.ceil(Ttotal/dec_factor)))
for i in range(59):
    
    x = pbs[i]
    random_phase = random.randint(0,dec_factor-1)
    indices = range(random_phase,Ttotal,int(dec_factor))
    PBS[i,0:len(indices)] = x[indices]
        
# Re-Estimate everything
Bsize_dec = PBS[:,0]
Bsize = pbs.iloc[0]
Tcycle_dec = np.zeros(59)
Tcycle = np.zeros(59)
for i in range(59):
    x = PBS[i,...]
    Tcycle_dec[i] = len(x[x>0]) * dec_factor
    x = pbs[i]
    Tcycle[i] = len(x[~np.isnan(x)])

plt.figure()
sb.regplot(Bsize,Tcycle * 10.0 / 60)
sb.regplot(Bsize_dec,Tcycle_dec * 10.0 / 60,scatter_kws={'s':40,'alpha':0.5})
plt.xlabel('Birth nuclear area (px)'),plt.xlim([0, 2500])
plt.ylabel('Total cell cycle duration (hr)')
Rcycle = stats.pearsonr( Bsize,Tcycle)
Rcycle_dec = stats.pearsonr( Bsize_dec,Tcycle_dec)

##########

# Plot the data
plt.subplot(1,2,1)
sb.regplot( pbs['Bsize'],pbs['Tg1'] )
plt.ylim([0,35])
plt.subplot(1,2,2)
sb.regplot( palbo['Bsize'],palbo['Tg1'] )
plt.ylim([0,35])

R_pbs = stats.pearsonr( pbs['Bsize'],pbs['Tg1'] )
R_palbo = stats.pearsonr( palbo['Bsize'],palbo['Tg1'] )

# Discretize Tg1 into %-iles
Tbins = np.linspace(0.,27., 10)
which_bin = np.digitize(pbs['Tg1'],Tbins)
pbs['Tg1 discrete'] = Tbins[which_bin-1]
which_bin = np.digitize(palbo['Tg1'],Tbins)
palbo['Tg1 discrete'] = Tbins[which_bin-1]


# Re-plot the data
plt.subplot(1,2,1)
sb.regplot( pbs['Bsize'],pbs['Tg1 discrete'] )
plt.subplot(1,2,2)
sb.regplot( palbo['Bsize'],palbo['Tg1 discrete'] )

R_pbs = stats.pearsonr( pbs['Bsize'],pbs['Tg1 discrete'] )
R_palbo = stats.pearsonr( palbo['Bsize'],palbo['Tg1 discrete'] )


