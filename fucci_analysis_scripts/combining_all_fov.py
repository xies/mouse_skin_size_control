#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 20:02:13 2019

@author: xies
"""

import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sb

# Combine all collated lists
collated = c1 + c2 + c5 + c6
regionID = [1,2,5,6]
Nregions = len(regionID)

# Put everything into a padded array for heatmap visualization
max_time = max( [len(c) for c in collated] )
Ncells = len(collated)

A = np.zeros((Ncells,max_time))
G1 = np.zeros((Ncells,max_time))

for (i,c) in enumerate(collated):
    a = c['ActinSegmentationArea']
    A[i,0:len(a)] = a/np.max(a)
    g1 = c['G1MarkerInActinSegmentationArea']
    g1 = g1 - g1.min()
    G1[i,0:len(a)] = g1/np.max(g1)
plt.subplot(1,2,1); plt.pcolor(A)
plt.subplot(1,2,2); plt.pcolor(G1)

# Check if G1marker signal is same in Voronoi v. segmented area
g1Voro = []
g1Actin = []
for c in collated:
    g1Voro = np.hstack( (g1Voro,np.array(c['G1MarkerInVoronoiArea'])) )
    g1Actin = np.hstack( (g1Actin,np.array(c['G1MarkerInActinSegmentationArea'])) )
sb.regplot(g1Voro,g1Actin) # They're not the same... use Actin!

# Call G1/S transition as when Cdt1 signal is max
for c in collated:
    age_at_g1s = (np.array(c['G1MarkerInVoronoiArea']).argmax()-0.0)/2.
    c['AgeAtG1S'] = age_at_g1s

# Plot birth size and T cell cycle
Tcycle = np.array([len(c)* 0.5 for c in collated])
Tg1 = np.array([np.array(c['AgeAtG1S'])[0] for c in collated])
Bsize = np.array([c['ActinSegmentationArea'].tolist()[0] for c in collated])
Region = np.array([np.unique(c['Region'])[0] for c in collated])


# Print the Pearson correlation coefficients (Region-specific)
Rg1 = np.zeros(Nregions)
Rcycle = np.zeros(Nregions)
for i in range(Nregions):
    I = Region == regionID[i]
    bsize = Bsize[I,...]
    tg1 = Tg1[I,...]
    tlength = Tcycle[I,...]
    Rg1[i] = stats.pearsonr(bsize,tg1)[0]
    Rcycle[i] = stats.pearsonr(bsize,tlength)[0]

# Look at basic linear regression
for i in range(Nregions):
    # Plot as scatter plot
    I = Region == regionID[i]
    plt.subplot(2,2,i+1)    
    # Overlay linear regression
    sb.regplot(Bsize[I,...],Tg1[I,...], color ='blue',marker='+',y_jitter=True)
    plt.xlabel('Cross-section area (px^2)')
    plt.ylabel('Cell cycle duration (days)')

# Look for binned birth size
plt.figure()
for i in range(Nregions):
    plt.subplot(2,2,i+1)
    I = Region == regionID[i]
    bsize = Bsize[I,...]
    tlength =Tcycle[I,...]
    # Plot scatter
    plt.scatter(bsize,jitter(tlength,.1),facecolor='none',edgecolor='r')
    # Find equal-size bins
    bin_edges = stats.mstats.mquantiles(bsize, [0, 1./8, 2./8, 3./8, 4./8, 5./8,6./8,7./8,1])
    Nbins = len(bin_edges) - 1
    
    which_bin = np.digitize(bsize,bin_edges)
    means = np.zeros(Nbins)
    stds = np.zeros(Nbins)
    bin_centers = np.zeros(Nbins)
    for b in range(Nbins):
        x = tlength[which_bin == b+1]
        bin_centers[b] = (bin_edges[b] + bin_edges[b+1]) / 2
        means[b] = x.mean()
        stds[b] = x.std() / np.sqrt(len(x))
    plt.errorbar(bin_centers,means,stds)
    plt.xlabel('Cross-section area (px^2)')
    plt.ylabel('Cell cycle duration (days)')
    
# Look at 2D histogram
plt.figure()
for i in range(Nregions):
    plt.subplot(2,2,i+1)
    I = Region == regionID[i]
    bsize = Bsize[I,...]
    tlength = Tcycle[I,...]
    # Plot scatter
    plt.hist2d(bsize,tlength,bins=[50,8])
    plt.colorbar()
    plt.xlabel('Cross-section area (px^2)')
    plt.ylabel('Cell cycle duration (days)')
    
#H, xedges, yedges = np.histogram2d(Bsize,Tcycle,bins=(15,15))
#X, Y = np.meshgrid(xedges, yedges)
#plt.pcolormesh(X, Y, H)
    

# Look at everything together
bsize = Bsize
tlength = jitter(Tcycle,.1)
# Plot scatter
plt.scatter(bsize,tlength,facecolor='none',edgecolor='r')
# Find equal-size bins
bin_edges = stats.mstats.mquantiles(bsize, [0, 1./8, 2./8, 3./8, 4./8, 5./8,6./8,7./8,1])
Nbins = len(bin_edges) - 1

which_bin = np.digitize(bsize,bin_edges)
means = np.zeros(Nbins)
stds = np.zeros(Nbins)
bin_centers = np.zeros(Nbins)
for b in range(Nbins):
    x = tlength[which_bin == b+1]
    bin_centers[b] = (bin_edges[b] + bin_edges[b+1]) / 2
    means[b] = x.mean()
    stds[b] = x.std() / np.sqrt(len(x))
plt.errorbar(bin_centers,means,stds)
plt.xlabel('Cross-section area (px^2)')
plt.ylabel('Cell cycle duration (days)')

## Censor Birth size outliers (5% top/bottom)
for i in range(Nregions):
    I = Region == regionID[i]
    bsize = Bsize[I,...]
    tlength = Tcycle[I,...]
    tg1 = Tg1[I,...]
    [mini,maxi] = stats.mstats.mquantiles(bsize,[.05,.95])
    filtered = (bsize > mini) & (bsize < maxi)
    plt.subplot(2,2,i+1)
    sb.regplot(bsize[filtered],tg1[filtered])
    plt.xlim([100,900]); plt.ylim([-.5,5])
    
    Rg1[i] = stats.pearsonr(bsize[filtered],tg1[filtered])[0]
    Rcycle[i] = stats.pearsonr(bsize[filtered],tlength[filtered])[0]
    
    
