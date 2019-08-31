#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:30:05 2019

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pickle as pkl


################################################
# Concatenate growth curves into heatmap
t = np.arange(15 - 3) * 12

# Heatmap
# Contatenate curves for heatmap
Ncells = len(collated_filtered)
[X,Y] = np.meshgrid(t,np.arange(Ncells))
GC = np.empty(X.shape) * np.nan
for i,c in enumerate(collated_filtered):
    V = c[c['Daughter'] == 'None'].Volume
    GC[i,0:len(V)] = V
    
##### Plot growth curve(s) without alignment
fig=plt.figure()
curve_colors = {'M1R1':'b','M1R2':'r','M2R5':'g'}
for c in collated:
    c = c[c['Daughter'] == 'None']
    v = np.array(c['Volume'],dtype=np.float)
    x = np.array(xrange(len(v))) * 12
    plt.plot(x,v,color=curve_colors[c.iloc[0].Region]) # growth curve
    if any(v>1000):
        print c
plt.xlabel('Time since birth (hr)')
plt.ylabel('Volume (um3)')

# Heatmap of birth-aligned growth curves
# Sort by length of cell cycle
t = np.arange(15 - 3) * 12
X,Y = np.meshgrid(t,np.arange(1,Ncells+1))
I = np.argsort(np.apply_along_axis(lambda x: len(nonans(x)),1,GC))
plt.pcolor(X,Y,GC[I,:])
plt.colorbar()
plt.xlabel('Time since birth (hr)')

####################################
# Plot histogram of fold grown
plt.hist(df['Fold grown'])
plt.vlines(df['Fold grown'].mean(),0,50)
plt.ylim([0,50])
plt.xlabel('Division volume / Birth volume')

# Plot histogram of fold grown
plt.hist(df['G1 grown'] / df['Total growth'])
plt.vlines((df['G1 grown'] / df['Total growth']).mean(),0,50)
plt.ylim([0,50])
plt.xlabel('G1 growth / Total growth')

####################################
# construct growth curves aligned at g1/s

g1lengths = pd.DataFrame([len(c[c['Phase'] == 'G1']) for c in collated])
sg2lengths = pd.DataFrame([len(c[c['Phase'] == 'SG2']) for c in collated])
phase_ambiguous = np.array([c['Phase'].iloc[0] == '?' for c in collated])
g1lengths[phase_ambiguous] = '?'
sg2lengths[phase_ambiguous] = '?'
Ncells = (g1lengths != '?').sum().values[0]

max_g1 = int(g1lengths[g1lengths != '?'].max().values[0])
max_sg2 = int(sg2lengths[g1lengths != '?'].max().values[0])

#initialize array with NaN
g1exit_aligned = np.empty((Ncells,max_g1+max_sg2)) * np.nan
g1exit_h2b = np.empty((Ncells,max_g1+max_sg2)) * np.nan
g1exit_fucci = np.empty((Ncells,max_g1+max_sg2)) * np.nan
g1exit_nuc = np.empty((Ncells,max_g1+max_sg2)) * np.nan

g1_aligned_frame = max_g1
g1_notaligned = np.empty((Ncells,10)) * np.nan

for i,c in enumerate(collated_filtered):
    this_phase = c['Phase']
    v = c[c.Daughter == 'None'].Volume
    g1_notaligned[i,0:len(v)] = v
    # Volume
    this_g1_vol = c[this_phase == 'G1']['Volume']
    this_sg2_vol = c[this_phase == 'SG2']['Volume']
    g1exit_aligned[i,g1_aligned_frame-len(this_g1_vol):g1_aligned_frame] = this_g1_vol
    g1exit_aligned[i,g1_aligned_frame:g1_aligned_frame+len(this_sg2_vol)] = this_sg2_vol
    # Nuclear volume
    this_g1_nuc = c[this_phase == 'G1']['Nucleus']
    this_sg2_nuc = c[this_phase == 'SG2']['Nucleus']
    g1exit_nuc[i,g1_aligned_frame-len(this_g1_vol):g1_aligned_frame] = this_g1_nuc
    g1exit_nuc[i,g1_aligned_frame:g1_aligned_frame+len(this_sg2_vol)] = this_sg2_nuc

    # FUCCI
    this_g1_fucci = c[this_phase == 'G1']['G1']
    this_sg2_fucci = c[this_phase == 'SG2']['G1']
    g1exit_fucci[i,g1_aligned_frame-len(this_g1_vol):g1_aligned_frame] = this_g1_fucci
    g1exit_fucci[i,g1_aligned_frame:g1_aligned_frame+len(this_sg2_vol)] = this_sg2_fucci

t = np.arange(-max_g1 + 1 ,max_sg2 + 1) * 12
t_birth = np.arange(10) * 12

####################################
#Plot aligned curves

X,Y = np.meshgrid(t_birth,np.arange(1,Ncells+1))
plt.pcolor(X,Y,g1_notaligned,cmap='inferno')
plt.xlabel('Time since birth (hr)')
plt.colorbar()

# Plot G1-aligned growth curves
X,Y = np.meshgrid(t,np.arange(1,Ncells + 1))
plt.pcolor(X,Y,g1exit_aligned,cmap='inferno') # Heatmap ->need to control meshgrid
plt.xlabel('Time since G1 exit (hr)')
plt.ylabel('Individual cells')
plt.colorbar

plt.figure()
colors = {'M1R1':'b','M1R2':'b','M2R5':'b'}
for i in xrange(Ncells):
    plt.plot(t,g1exit_aligned[i,:],
    color=colors[collated_filtered[i].iloc[0].Region],alpha=0.2)
# plot mean/error as shade
Ncell_in_bin = (~np.isnan(g1exit_aligned)).sum(axis=0)
mean_curve = np.nanmean(g1exit_aligned,axis=0)
mean_curve[Ncell_in_bin < 10] = np.nan
std_curve = np.nanstd(g1exit_aligned,axis=0)
std_curve[Ncell_in_bin < 10] = np.nan
plt.plot(t, mean_curve, color='r')
#plt.fill_between(t, mean_curve-std_curve, mean_curve+std_curve,
#                 color='k',alpha=0.5)
plt.xlabel('Time since G1 exit (hr)')
plt.ylabel('Cell volume (um3)')

# Boxplots
mask = ~np.isnan(g1exit_aligned)
filtered_data = [d[m] for d, m in zip(g1exit_aligned.T, mask.T)]
plt.boxplot(filtered_data)


# Plot G1-aligned CV
plt.figure()
t = np.arange(-max_g1 + 1 ,max_sg2 + 1) * 12
# plot mean/error as shade
Ncell_in_bin = (~np.isnan(g1exit_aligned)).sum(axis=0)
cv = stats.variation(g1exit_aligned,axis=0,nan_policy='omit')
cv[Ncell_in_bin < 10] = np.nan
plt.plot(t,cv)

################################################
# Plot nuclear growth
plt.figure()

colors = {'M1R1':'b','M1R2':'b','M2R5':'b'}
for i in xrange(Ncells):
    plt.plot(t,g1exit_nuc[i,:],
             colors[collated_filtered[i].iloc[0].Region],alpha=0.2)
    
Ncell_in_bin = (~np.isnan(g1exit_nuc)).sum(axis=0)
mean_curve = np.nanmean(g1exit_nuc,axis=0)
mean_curve[Ncell_in_bin < 10] = np.nan
std_curve = np.nanstd(g1exit_nuc,axis=0)
std_curve[Ncell_in_bin < 10] = np.nan
plt.plot(t, mean_curve, color='r')
plt.fill_between(t, mean_curve-std_curve, mean_curve+std_curve,
                 color='k',alpha=0.5)
plt.plot(t, mean_curve, color='r')
plt.xlabel('Time since G1 exit (hr)')
plt.ylabel('Nuclear volume (um3)')


# Plot N:C ratio wrt time
plt.figure()
for i in xrange(Ncells):
    plt.plot(t,g1exit_nuc[i,:]/g1exit_aligned[i,:],
             colors[collated_filtered[i].iloc[0].Region],alpha=0.2)
Ncell_in_bin = (~np.isnan(g1exit_nuc)).sum(axis=0)
mean_curve = np.nanmean(g1exit_nuc/g1exit_aligned,axis=0)
mean_curve[Ncell_in_bin < 10] = np.nan
std_curve = np.nanstd(g1exit_nuc/g1exit_aligned,axis=0)
std_curve[Ncell_in_bin < 10] = np.nan
plt.plot(t, mean_curve, color='r')
plt.fill_between(t, mean_curve-std_curve, mean_curve+std_curve,
                 color='k',alpha=0.5)
plt.plot(t, mean_curve, color='r')
plt.xlabel('Time since G1 exit (hr)')
plt.ylabel('Nuclear : cytoplasmic ratio')


# Plot N:C ratio wrt size
for c in collated_filtered:
    c = c[c['Daughter'] == 'None']
    plt.scatter(c['Volume (sm)'],c.Nucleus/c['Volume (sm)'],color='b')
Ncell_in_bin = (~np.isnan(g1exit_nuc)).sum(axis=0)
mean_curve = np.nanmean(g1exit_nuc/g1exit_aligned,axis=0)
mean_curve[Ncell_in_bin < 10] = np.nan
std_curve = np.nanstd(g1exit_nuc/g1exit_aligned,axis=0)
std_curve[Ncell_in_bin < 10] = np.nan
plt.plot(t, mean_curve, color='r')
#plt.fill_between(t, mean_curve-std_curve, mean_curve+std_curve,
#                 color='k',alpha=0.5)
plt.plot(t, mean_curve, color='r')
plt.xlabel('Time since G1 exit (hr)')
plt.ylabel('Nuclear : cytoplasmic ratio')


V = np.hstack([c[c['Daughter'] == 'None']['Volume (sm)'].values for c in collated_filtered])
nV = np.hstack([c[c['Daughter'] == 'None'].Nucleus.values for c in collated_filtered])
phases = np.hstack([c[c['Daughter'] == 'None'].Phase.values for c in collated_filtered])

bins = stats.mstats.mquantiles(x['Volume'],np.array([0,1.,2.,3.,4.,5.,6.,7.])/7)
x = pd.DataFrame(np.vstack((V,nV/V)).T,columns=['Volume','Ratio'])
x['Phase'] = phases
sb.lmplot(data = x, x='Volume',y='Ratio',hue='Phase',fit_reg=False)
plot_bin_means(x['Volume'],x['Ratio'],bins,color='r',error='std',style='fill')
#sb.regplot(data = x, x='Volume',y='Ratio',scatter=False)
plt.xlabel('Cell volume (um3)')
plt.ylabel('N:C ratio')



