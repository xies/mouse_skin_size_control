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

plt.figure()
plt.hist(nonans(df_has_daughter['Daughter ratio']))
plt.vlines(np.nanmean(df_has_daughter['Daughter ratio']),0,50)
plt.xlabel('Daughter volume ratio')
plt.ylabel('Frequency')

# Histogram before and after
mitotic = df_has_daughter[df_has_daughter.Mitosis]
non_mitotic = df_has_daughter[~df_has_daughter.Mitosis]
plt.figure()
plt.hist( non_mitotic['Division volume'] ,histtype='step')
plt.hist( non_mitotic['Division volume interpolated'] ,histtype='step')
plt.vlines([np.nanmean(non_mitotic['Division volume']),
            np.nanmean(non_mitotic['Division volume interpolated'])],0,50)
plt.xlabel('Division volume (um3)')

plt.figure()
plt.hist( mitotic['Division volume'],histtype='step')
plt.hist( mitotic['Division volume interpolated'],histtype='step')
plt.vlines([np.nanmean(mitotic['Division volume']),
            np.nanmean(mitotic['Division volume interpolated'])],0,3)
plt.xlabel('Division volume (um3)')

# Histogram difference
plt.figure()
plt.hist( nonans( (df_has_daughter['Division volume interpolated']-df_has_daughter['Division volume']) * 100 /
                 df_has_daughter['Division volume'] ))
plt.xlabel('% difference between final volume and sum of daughter volumes')
plt.vlines(np.mean(nonans( (df_has_daughter['Division volume interpolated']-df_has_daughter['Division volume']) * 100 /
                 df_has_daughter['Division volume'] )),0,30)
# Plot daughter 'ghost' growth point
mitosis_color = {True:'o',False:'b'}
has_daughter = [c for c in collated if np.any(c.Daughter != 'None')]


########################################################################
# Quantify as slope plot

X = df_has_daughter.iloc[df_has_daughter.groupby('Mitosis').indices[False]]['Division volume'].values
Y = df_has_daughter.iloc[df_has_daughter.groupby('Mitosis').indices[False]]['Division volume interpolated'].values
plot_slopegraph(X,Y,names=['Final volume','Interpolated final volume'])

plt.plot([1,2],[X.mean(),Y.mean()],color='r')
plt.xlim([0,3])


X = df_has_daughter.iloc[df_has_daughter.groupby('Mitosis').indices[True]]['Division volume'].values
Y = df_has_daughter.iloc[df_has_daughter.groupby('Mitosis').indices[True]]['Division volume interpolated'].values
Y = Y + (Y-X)
plot_slopegraph(X,Y,names=['Final volume','Total daughter volume'],color='orange')

plt.errorbar([1,2],[X.mean(),Y.mean()], yerr = [stats.sem(X),stats.sem(Y)],color='k')
plt.xlim([0,3])


# Division differences
df_has_daughter['Division difference'] = df_has_daughter['Division volume interpolated'] - df_has_daughter['Division volume']

sb.violinplot(data=df_has_daughter,y='Division difference',x='Mitosis')
plt.xlabel('Fold grown interpolated')


# Fold-growth
plt.figure()
plt.hist( nonans(df['Division volume interpolated']/df['Birth volume']) )
plt.vlines([nonans(df['Division volume interpolated']/df['Birth volume']).mean(),
            df['Fold grown'].mean() ],0,30)
plt.xlabel('Fold grown interpolated')

plt.figure()
plt.hist( nonans( (df['Division volume interpolated']-df['Division volume'])/df['Division volume']) )
plt.vlines(nonans( (df['Division volume interpolated']-df['Division volume'])/df['Division volume']).mean(),0,30)
plt.xlabel('Diff')

########################################################################
    


