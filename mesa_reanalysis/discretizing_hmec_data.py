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

dt = 1./6
deci_factor = 35

filename = '/Users/xies/Box/HMECs/HMEC DFB tracked data/Old dataset/pbs_growth.csv'
pbs_growth_curves = pd.read_csv(filename,names=range(59))
filename = '/Users/xies/Box/HMECs/HMEC DFB tracked data/Old dataset/pbs.csv'
pbs = pd.read_csv(filename,header=None).T
pbs.columns = ['Birth size','G1 length']
pbs['G1 frame'] = np.round(pbs['G1 length'].values * 6)
g1_frame = pbs['G1 frame'].astype(np.int)
Ncells = len(pbs)

###### Collate data
Ttotal = len(pbs_growth_curves[0])
dsize = np.zeros(Ncells)
bsize = np.zeros(Ncells)
bsize_ = np.zeros(Ncells)
g2len = np.zeros(Ncells)
g1size = np.zeros(Ncells)
totalT = np.zeros(Ncells)
for i in range(Ncells):
    x = nonans(pbs_growth_curves[i].values)
    Tx = len(x)
    t = range(Tx)
    # Grab birth size
    bsize[i] = x[0]
    # Grab size at G1
    g1size[i] = x[g1_frame[i]]
    # Grab division size
    dsize[i] = x[-1]
    # Grab total cell cycle length
    totalT[i] = Tx * dt
    # Grab SG2M duration
    g2len[i] = (Tx - g1_frame[i]) * dt

pbs['Division size'] = dsize
pbs['Birth size'] = bsize
pbs['G1 size'] = g1size
pbs['SG2 length'] = g2len
pbs['Total length'] = totalT
pbs['G1 growth'] = pbs['G1 size'] - pbs['Birth size']
pbs['SG2 growth'] = pbs['Division size'] - pbs['G1 size']
pbs['Total growth'] = pbs['Division size'] - pbs['Birth size']

R_g1_len = np.corrcoef(pbs['Birth size'],pbs['G1 length'])[0,1]
R_g2_len = np.corrcoef(pbs['G1 size'],pbs['SG2 length'])[0,1]
R_g1_growth = np.corrcoef(pbs['Birth size'],pbs['G1 growth'])[0,1]
R_g2_growth = np.corrcoef(pbs['G1 size'],pbs['SG2 growth'])[0,1]
R_total_growth = np.corrcoef(pbs['Birth size'],pbs['Total growth'])[0,1]

# Collect statistics

Niter = 500
R_g1_len_ = np.empty(Niter) * np.nan
R_g2_len_ = np.empty(Niter) * np.nan
R_g1_growth_ = np.empty(Niter) * np.nan
R_g2_growth_ = np.empty(Niter) * np.nan
R_total_growth_ = np.empty(Niter) * np.nan
for i in range(Niter):
    # Generate decimated data
    try:
        pbs_decimated = decimate_data(pbs_growth_curves,pbs,deci_factor,dt)
    
        df = pbs.join(pbs_decimated,lsuffix='',rsuffix=' decimated')
        birth_vol_bins = stats.mstats.mquantiles(df['Birth size'], [0, 1./6, 2./6, 3./6, 4./6, 6./6, 1])
        g1_vol_bins = stats.mstats.mquantiles(df['G1 size'], [0, 1./6, 2./6, 3./6, 4./6, 6./6, 1])
        
        R_g1_len_[i] = np.corrcoef(pbs_decimated['Birth size'],pbs_decimated['G1 length'])[0,1]
        R_g2_len_[i] = np.corrcoef(pbs_decimated['G1 size'],pbs_decimated['SG2 length'])[0,1]
        R_g1_growth_[i] = np.corrcoef(pbs_decimated['Birth size'],pbs_decimated['G1 growth'])[0,1]
        R_g2_growth_[i] = np.corrcoef(pbs_decimated['G1 size'],pbs_decimated['SG2 growth'])[0,1]
        R_total_growth_[i] = np.corrcoef(pbs_decimated['Birth size'],pbs_decimated['Total growth'])[0,1]
        
    ####### Plot decimated data
        plt.figure(1)
        plot_size_v_duration(pbs_decimated,color='k',scatter_kws={'alpha':0.1})
        plt.figure(2)
        plot_size_v_growth(pbs_decimated,color='k',scatter_kws={'alpha':0.1})
        plt.figure(3)
        plot_overall_growth(pbs_decimated,color='k',scatter_kws={'alpha':0.1})
            
    except:
        print 'decimation failed'

# Plot real data
plt.figure(1)
plot_size_v_duration(pbs,color='r')
plt.figure(2)
plot_size_v_growth(pbs,color='r')
plt.figure(3)
plot_overall_growth(pbs,color='r')

##########

plt.figure()
plt.hist(nonans(R_g1_len_))
plt.hist(nonans(R_g2_len_))
plt.legend(('G1','S/G2/M'))
plt.vlines((R_g1_len,R_g2_len),ymin=0,ymax=120)
plt.xlabel('Correlation between entry size and phase duration')

plt.figure()
plt.hist(nonans(R_g1_growth_),histtype='step')
plt.hist(nonans(R_g2_growth_),histtype='step')
plt.hist(nonans(R_total_growth_),histtype='step')
plt.legend(('G1','S/G2/M','Total'))
plt.vlines((R_g1_growth,R_g2_growth,R_total_growth),ymin=0,ymax=120)
#plt.ylim([0,30])
plt.xlabel('Correlation between entry size and growth during indicated phase')

########
# Change decimation factor and see how correlation distribution changes

Niter = 500
decimations2try = [10,20,30,35]
decimation_examples = []

R_g2_len_ = np.empty((4,Niter)) * np.nan
for j,deci in enumerate(decimations2try):

    counter = 0
    for i in range(Niter):
        # Generate decimated data
        try:
            pbs_decimated = decimate_data(pbs_growth_curves,pbs,deci,dt)                 
            R_g2_len_[j,i] = np.corrcoef(pbs_decimated['G1 size'],pbs_decimated['SG2 length'])[0,1]
            counter += 1
            if counter == 1:
                decimation_examples.append(pbs_decimated)
        except:
            print 'failed'

for i in range(4):
    x = nonans(R_g2_len_[i,:])
    weights = np.ones_like(x)/float(len(x))
    plt.hist( x,histtype='step',density=False,weights=weights)
plt.vlines((R_g2_len),ymin=0,ymax=.3)
plt.xlabel('Correlation between G1 exit size and S/G2 duration')
plt.legend(decimations2try)

##### Histogram of phase durations
bins = np.arange(0,180/6.,15/6.)
plt.hist( pbs_decimated['G1 length'] ,histtype='step',stacked=True)
plt.hist( (pbs_decimated['Total length'] - pbs_decimated['G1 length']),bins = bins,histtype='step',stacked=True)
plt.xlabel('Phase duration (hr)')
plt.ylabel('Frequency')
plt.legend(['G1','S/G2/M'])


##########


def decimate_data(growth_curves,df,dec_factor,dt):
    ###### Need to decimate data by factor of 20 (with random phases)
    Ttotal = len(growth_curves[0])
    g1len_ = np.zeros(Ncells)
    dsize_ = np.zeros(Ncells)
    bsize_ = np.zeros(Ncells)
    g1frame_ = np.zeros(Ncells,dtype=np.int)
    g2len_ = np.zeros(Ncells)
    g1size_ = np.zeros(Ncells)
    totalT_ = np.zeros(Ncells)
    for i in range(Ncells):
        x = nonans(growth_curves[i].values)
        Tx = len(x)
        # Geberate a random decimation phase
        random_phase = random.randint(0,dec_factor-1)
        # Decimate
        indices = range(random_phase,Tx,int(dec_factor))
        x_decimated = x[indices]
        # Grab the nearest G1 frame estimate as last decimated frame before "real" G1/S
        g1frame_[i] = int(np.where(df.iloc[i]['G1 frame'] > indices)[0].max() + 1)
        g1len_[i] = g1frame_[i]* dt * dec_factor
        # Grab birth size
        bsize_[i] = x_decimated[0]
        # Grab size at G1
        g1size_[i] = x_decimated[g1frame_[i]]
        # Grab division size
        dsize_[i] = x_decimated[-1]
        # Grab total cell cycle length
        totalT_[i] = len(indices) * dt * dec_factor
        # Grab SG2M duration
        g2len_[i] = (len(indices) - g1frame_[i]) * dec_factor * dt
    
    # Collate decimated data
    decimated = pd.DataFrame()
    decimated['G1 frame'] = g1frame_
    decimated['Birth size'] = bsize_
    decimated['Division size'] = dsize_
    decimated['G1 size'] = g1size_
    decimated['SG2 length'] = g2len_
    decimated['G1 length'] = g1len_
    decimated['Total length'] = totalT_
    decimated['G1 growth'] = decimated['G1 size'] - decimated['Birth size']
    decimated['SG2 growth'] = decimated['Division size'] - decimated['G1 size']
    decimated['Total growth'] = decimated['Division size'] - decimated['Birth size']

    return decimated
    
    
def plot_size_v_duration(df,color='k',scatter_kws={}):
    ####### Analyze for phase duration
    birth_vol_bins = stats.mstats.mquantiles(df['Birth size'], [0, 1./6, 2./6, 3./6, 4./6, 6./6, 1])
    g1_vol_bins = stats.mstats.mquantiles(df['G1 size'], [0, 1./6, 2./6, 3./6, 4./6, 6./6, 1])
    # G1
    plt.subplot(2,1,1)
#    sb.regplot(data=df, x = 'Birth size decimated', y = 'G1 length decimated', fit_reg=False,color='b')
#    plot_bin_means(df['Birth size decimated'],df['G1 length decimated'],birth_vol_bins)
#    # SG2
#    plt.subplot(2,1,2)
#    sb.regplot(data=df, x = 'G1 size decimated', y = 'SG2 length decimated', fit_reg=False,color='b')
#    plot_bin_means(df['G1 size decimated'],df['SG2 length decimated'],g1_vol_bins)

    plt.subplot(2,1,1)
    sb.regplot(data=df, x = 'Birth size', y = 'G1 length', fit_reg=False, color=color,
               scatter_kws=scatter_kws, y_jitter=True)
    plot_bin_means(df['Birth size'],df['G1 length'],birth_vol_bins,color=color)
    plt.subplot(2,1,2)
    sb.regplot(data=df, x = 'G1 size', y = 'SG2 length', fit_reg=False, color=color,
               scatter_kws=scatter_kws, y_jitter=True)
    plot_bin_means(df['G1 size'],df['SG2 length'],g1_vol_bins,color=color)

def plot_size_v_growth(df,color='k',scatter_kws={}):
    ####### Analyze for phase growth
    birth_vol_bins = stats.mstats.mquantiles(df['Birth size'], [0, 1./6, 2./6, 3./6, 4./6, 6./6, 1])
    g1_vol_bins = stats.mstats.mquantiles(df['G1 size'], [0, 1./6, 2./6, 3./6, 4./6, 6./6, 1])
    # G1
#    plt.subplot(2,1,1)
#    sb.regplot(data=df, x = 'Birth size decimated', y = 'G1 growth decimated', fit_reg=False,color='b')
#    plot_bin_means(df['Birth size decimated'],df['G1 growth decimated'],birth_vol_bins)
#    # SG2
#    plt.subplot(2,1,2)
#    sb.regplot(data=df, x = 'G1 size decimated', y = 'SG2 growth decimated', fit_reg=False,color='b')
#    plot_bin_means(df['G1 size decimated'],df['SG2 growth decimated'],g1_vol_bins)
    
    plt.subplot(2,1,1)
    sb.regplot(data=df, x = 'Birth size', y = 'G1 growth', fit_reg=False, color=color,
               scatter_kws=scatter_kws)
    plot_bin_means(df['Birth size'],df['G1 growth'],birth_vol_bins,color=color)
    plt.subplot(2,1,2)
    sb.regplot(data=df, x = 'G1 size', y = 'SG2 growth', fit_reg=False, color=color,
               scatter_kws=scatter_kws)
    plot_bin_means(df['G1 size'],df['SG2 growth'],g1_vol_bins,color=color)
    

def plot_overall_growth(df,color='k',scatter_kws={}):
    ####### Analyze for overall growth
    birth_vol_bins = stats.mstats.mquantiles(df['Birth size'], [0, 1./6, 2./6, 3./6, 4./6, 6./6, 1])
    
    sb.regplot(data=df, x = 'Birth size', y = 'Total growth', fit_reg=False, color=color,
               scatter_kws=scatter_kws, y_jitter=True)
    plot_bin_means(df['Birth size'],df['Total growth'],birth_vol_bins,color=color)


    