#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:24:43 2019

@author: xies
"""

from numpy import random
import numpy as np
import matplotlib.pylab as plt
import seaborn as sb
from scipy import stats

ucellIDs = np.array([c.iloc[0].CellID for c in collated])

########################################################################

# Plot daughter division asymmetry
df_has_daughter = df[~np.isnan(df['Daughter a volume'])]

plt.hist(nonans(df_has_daughter['Daughter ratio']))
plt.vlines(np.nanmean(df_has_daughter['Daughter ratio']),0,30)
plt.xlabel('Daughter volume ratio')
plt.ylabel('Frequency')

# Histogram fold change
plt.hist( nonans( (df_has_daughter['Division volume interpolated']-df_has_daughter['Division volume']) * 100 /
                 df_has_daughter['Division volume'] ))
plt.xlabel('% difference between final volume and sum of daughter volumes')
plt.vlines(np.mean(nonans( (df_has_daughter['Division volume interpolated']-df_has_daughter['Division volume']) * 100 /
                 df_has_daughter['Division volume'] )),0,30)
# Plot daughter 'ghost' growth point
mitosis_color = {True:'o',False:'b'}
has_daughter = [c for c in collated if np.any(c.Daughter != 'None')]

Ncells = len(has_daughter)
GC_ = np.zeros((Ncells,13))
idx = random.randint(0,len(has_daughter),size=(10))
for i in idx:
    # Collate into heatmap
    c = has_daughter[i]
    mainC = c[c['Daughter'] == 'None']
    daughters = c[c['Daughter'] != 'None']
    t = (mainC.Frame - mainC.iloc[0].Frame)*12
    plt.plot(t,mainC.Volume,mitosis_color[c.iloc[-1]['Phase'] == 'M'])
    plt.plot([t.iloc[-1], t.iloc[-1] + 6],
         [mainC.iloc[-1].Volume,daughters.Volume.sum()],
         marker='o',linestyle='dashed',color='r')
    plt.xlabel('Time since birth (hr)')
    plt.ylabel('Cell volume')
    
    GC_[i,0:len(mainC)] = mainC.Volume
    GC_[i,len(mainC):len(mainC) + 1] = daughters.Volume.sum()

########################################################################
# Quantify as slope plot

X = df_has_daughter.iloc[df_has_daughter.groupby('Mitosis').indices[False]]['Division volume'].values
Y = df_has_daughter.iloc[df_has_daughter.groupby('Mitosis').indices[False]]['Division volume interpolated'].values
plot_slopegraph(X,Y,names=['Final volume','Interpolated final volume'])

plt.plot([1,2],[X.mean(),Y.mean()],color='r')
plt.xlim([0,3])


X = df_has_daughter.iloc[df_has_daughter.groupby('Mitosis').indices[True]]['Division volume'].values
Y = df_has_daughter.iloc[df_has_daughter.groupby('Mitosis').indices[True]]['Division volume interpolated'].values
plot_slopegraph(X,Y,names=['Final volume','Interpolated final volume'],color='orange')

plt.plot([1,2],[X.mean(),Y.mean()],color='k')
plt.xlim([0,3])

########################################################################
    
# Align growth curves at G1/S
collated_filtered = [c for c in has_daughter if c.iloc[0].Phase != '?']

g1lengths = pd.DataFrame([len(c[c['Phase'] == 'G1']) for c in collated_filtered])
sg2lengths = pd.DataFrame([len(c[c['Phase'] == 'SG2']) for c in collated_filtered])
phase_ambiguous = np.array([c['Phase'].iloc[0] == '?' for c in collated_filtered])
g1lengths[phase_ambiguous] = '?'
sg2lengths[phase_ambiguous] = '?'
Ncells = (g1lengths != '?').sum().values[0]

max_g1 = int(g1lengths[g1lengths != '?'].max().values[0])
max_sg2 = int(sg2lengths[g1lengths != '?'].max().values[0])

#initialize array with NaN
g1exit_aligned = np.empty((Ncells,max_g1+max_sg2)) * np.nan
g1_aligned_frame = max_g1
g1_notaligned = np.empty((Ncells,10)) * np.nan

for i,c in enumerate(collated_filtered):
    this_phase = c['Phase']
    v = c[c.Daughter == 'None'].Volume
    daughters = c[c['Daughter'] != 'None']
    # Volume
    this_g1_vol = c[this_phase == 'G1']['Volume']
    this_sg2_vol = c[this_phase == 'SG2']['Volume']
    daughter_vol = c[this_phase == 'Daughter G1'].Volume.sum()
    g1exit_aligned[i,g1_aligned_frame-len(this_g1_vol):g1_aligned_frame] = this_g1_vol
    g1exit_aligned[i,g1_aligned_frame:g1_aligned_frame+len(this_sg2_vol)] = this_sg2_vol
    g1exit_aligned[i,g1_aligned_frame+len(this_sg2_vol):g1_aligned_frame+len(this_sg2_vol)+1] = daughter_vol

t = np.arange(-max_g1 + 1 ,max_sg2 + 1) * 12
t_birth = np.arange(10) * 12


# Heatmap 
X,Y = np.meshgrid(t,range(Ncells))
plt.pcolor(X,Y,g1exit_aligned)

########################################################################
# Calculate size correlations

# Construct histogram bins
nbins = 5
birth_vol_bins = stats.mstats.mquantiles(df['Birth volume'],  np.arange(0,nbins+1,dtype=np.float)/nbins)
g1_vol_bins = stats.mstats.mquantiles(df['G1 volume'], np.arange(0,nbins+1,dtype=np.float)/nbins)

####### Corrected overall correlations
# 1) Volume
sb.regplot(data=df,x='Birth volume',y='Division volume',fit_reg=False,ci=None,scatter=False)
plot_bin_means(df['Birth volume'],df['Division volume'],birth_vol_bins,color='blue')
sb.regplot(data=df,x='Birth volume',y='Division volume interpolated',fit_reg=True,ci=None)
plot_bin_means(df['Birth volume'],df['Division volume interpolated'],birth_vol_bins,color='green')
plt.xlabel('Birth volume (um3)')
plt.ylabel('Division volume (interpolated) (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([200,550])
plt.legend(['Original data','Daughter volume interpolation'])

# Linear regression
x = df['Birth volume']
y = df['Division volume interpolated']
I = ~(np.isnan(y) | np.isnan(x))
Pdivinterp = np.polyfit(x[I],y[I],1)
print 'Slope of final division: ', Pdivinterp[0]
# Pearson
Rdivinterp,P = stats.stats.pearsonr(x[I],y[I])
print 'Correlation of final division: ', Rdivinterp,P

## Corrected overall correlations
# 2) Growth
sb.regplot(data=df,x='Birth volume',y='Total growth',fit_reg=True,ci=None,scatter=True)
#plot_bin_means(df['Birth volume'],df['Division volume'],birth_vol_bins,color='blue')
sb.regplot(data=df,x='Birth volume',y='Total growth interpolated',fit_reg=True,ci=None)
plot_bin_means(df['Birth volume'],df['Total growth interpolated'],birth_vol_bins,color='green')
plt.xlabel('Birth volume (um3)')
plt.ylabel('Division volume (interpolated) (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([200,550])
plt.legend(['Original data','Daughter volume interpolation'])

# Linear regression
x = df['Birth volume']
y = df['Division volume interpolated']-df['Birth volume']
I = ~(np.isnan(y) | np.isnan(x))
Pgrowthinterp = np.polyfit(x[I],y[I],1)
print 'Slope of total growth: ', Pgrowthinterp[0]
# Pearson
Rgrowthinterp,P = stats.stats.pearsonr(x[I],y[I])
print 'Correlation of total growth: ', Rgrowthinterp,P


####### Corrected S/G2 correlations
# 1) Volume
sb.lmplot(data=df,x='G1 volume',y='Division volume',fit_reg=True,ci=None)
plot_bin_means(df['G1 volume'],df['Division volume'],g1_vol_bins)
sb.regplot(data=df,x='G1 volume',y='Division volume interpolated',fit_reg=True,ci=None)
plot_bin_means(df['G1 volume'],df['Division volume interpolated'],g1_vol_bins,color='green')
plt.xlabel('G1 exit volume (um3)')
plt.ylabel('Division volume (interpolated) (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([250,650])
plt.legend(['Original data','Daughter volume interpolation'])


# Linear regression
I = ~np.isnan(df['Division volume interpolated'])
Pg1pdivinterp = np.polyfit(df.loc[I,'G1 volume'],df.loc[I,'Division volume interpolated'],1)
print 'Slope of G1 growth: ', Pg1pdivinterp[0]


sb.lmplot(data=df,x='G1 volume',y='Division volume',fit_reg=True,ci=None)
plot_bin_means(df['G1 volume'],df['Division volume'],g1_vol_bins)
sb.regplot(data=df,x='G1 volume interpolated',y='Division volume interpolated',fit_reg=True,ci=None)
plot_bin_means(df['G1 volume interpolated'],df['Division volume interpolated'],g1_vol_bins,color='green')
plt.xlabel('G1 exit volume (um3)')
plt.ylabel('Division volume (interpolated) (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([250,650])
plt.legend(['Original data','Interpolation'])

# Double-corrected S/G2 correlation
# Growth
sb.lmplot(data=df,x='G1 volume',y='SG2 grown',fit_reg=True,ci=None)
#plot_bin_means(df['G1 volume'],df['SG2 grown'],g1_vol_bins)
sb.regplot(df['G1 volume interpolated'],df['Division volume interpolated']-df['G1 volume interpolated'],fit_reg=True,ci=None,color='orange')
plot_bin_means(df['G1 volume interpolated'],df['Division volume interpolated']-df['G1 volume interpolated'],g1_vol_bins)
plt.xlabel('G1 exit volume (um3)')
plt.ylabel('SG2 growth (interpolated) (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([250,650])
plt.legend(['Original data','Interpolation'])


# Linear regression
I = ~( np.isnan(df['Division volume interpolated']) | np.isnan(df['G1 volume interpolated']))
Pdivinterp = np.polyfit(df.loc[I,'G1 volume interpolated'],df.loc[I,'Division volume interpolated'],1)
print 'Slope of G1 growth: ', Pdivinterp[0]
# Pearson
I = ~(np.isnan(df['G1 volume interpolated']) | np.isnan(df['Division volume interpolated']))
Rg1volinterp,P = stats.stats.pearsonr(df.loc[I,'G1 volume interpolated'],df.loc[I,'Division volume interpolated'])
print 'Correlation of G1 length: ', Rg1volinterp,P

# Final volume
sb.lmplot(data=df,x='G1 volume',y='Division volume',fit_reg=True,ci=None,scatter=False)
#plot_bin_means(df['G1 volume'],df['SG2 grown'],g1_vol_bins)
sb.regplot(df['G1 volume interpolated'],df['Division volume interpolated'],fit_reg=True,ci=None)
plot_bin_means(df['G1 volume interpolated'],df['Division volume interpolated'],g1_vol_bins,color='green')
plt.xlabel('G1 exit volume (interpolated) (um3)')
plt.ylabel('S/G2 volume (interpolated) (um3)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([250,650])
plt.legend(['Original data','Interpolation'])


# Linear regression
I = ~( np.isnan(df['Division volume interpolated']) | np.isnan(df['G1 volume interpolated']))
Pdivinterp = np.polyfit(df.loc[I,'G1 volume interpolated'],df.loc[I,'Division volume interpolated']-df.loc[I,'G1 volume interpolated'],1)
print 'Slope of G1 growth: ', Pdivinterp[0]

# Pearson
I = ~(np.isnan(df['G1 volume interpolated']) | np.isnan(df['Division volume interpolated']))
Rg1volinterp,P = stats.stats.pearsonr(df.loc[I,'Birth volume'],df.loc[I,'Division volume interpolated']-df.loc[I,'G1 volume interpolated'])
print 'Correlation of G1 length: ', Rg1volinterp,P




