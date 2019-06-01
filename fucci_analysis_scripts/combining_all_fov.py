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
import pickle
plt.rcParams.update({'font.size': 16})

# Combine all collated lists
collated = c1 + c2 + c5 + c6
#collated = f1 + f2 + f5 + f6
regionID = [1,2,5,6]
Nregions = len(regionID)

# Save collated as bitstream
output = open('/Users/mimi/Box Sync/Mouse/Skin/collated.pkl','wb')
pickle.dump(collated, output)
output.close()

# Put everything into a padded array for heatmap visualization
max_time = max( [len(c) for c in collated] )
Ncells = len(collated)

A = np.empty((Ncells,max_time)) * np.nan
G1 = np.empty((Ncells,max_time)) * np.nan
for (i,c) in enumerate(collated):
    a = c['ActinSegmentationArea']
    A[i,0:len(a)] = a / a.mean()
    g1 = c['G1MarkerInVoronoiArea']
    G1[i,0:len(a)] = g1 / g1.mean()

    
# Plot heatmaps
plt.clf()
plt.subplot(1,2,1)
plt.pcolor(A,vmin=0,vmax=2,cmap='inferno')
plt.xlabel('Half-days')
plt.ylabel('Individual cells')
plt.title('Crosssection area normalized\n by single-cell-mean (px2)')
plt.colorbar()
plt.subplot(1,2,2)
plt.pcolor(G1,vmin=0,vmax=2,cmap='inferno')
plt.xlabel('Half-days')
plt.ylabel('Individual cells')
plt.title('FUCCI-G1 signal normalized\n by single-cell-mean (au)')
plt.colorbar()

# Check when G1max occurs WRT time of division
whenG1max = np.zeros(Ncells)
for (i,c) in enumerate(collated):
    divFrame = len(c)
    g1max = np.array(c['G1MarkerInVoronoiArea']).argmax()
    whenG1max[i] = (g1max - divFrame)*0.5
weights = np.ones_like(whenG1max) / float(len(whenG1max))
plt.figure()
plt.hist(whenG1max,len(np.unique(whenG1max)), weights=weights)
plt.xlabel('Days from division')
plt.title('Timing when FUCCI-G1\n is maximum WRT division')

# Check when AreaMax occurs WRT time of division
whenAreamax = np.zeros(Ncells)
for (i,c) in enumerate(collated):
    divFrame = len(c)
    areamax = np.array(c['ActinSegmentationArea']).argmax()
    whenAreamax[i] = (areamax - divFrame)*0.5
plt.figure()
weights = np.ones_like(whenAreamax) / float(len(whenAreamax))
plt.hist(whenAreamax,len(np.unique(whenG1max)),weights=weights)
plt.xlabel('Days from division')
plt.title('Timing when cell area\n is maximum WRT division')

 
# Check if Areadt is mostly positive (NTS: it's not)
dAreadt = np.diff(A,n=1)
plt.hist( nonans(dAreadt.flatten('F')),100,normed=True)
plt.axvline(x=0,color='r')
plt.xlabel('dArea/dt')

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
    c['G1Growth'] = c['ActinSegmentationArea'].tolist()[np.array(c['G1MarkerInVoronoiArea']).argmax()] - \
            c['ActinSegmentationArea'].tolist()[0]

# Plot birth size and T cell cycle
Tcycle = np.array([len(c)* 0.5 for c in collated])
Tg1 = np.array([np.array(c['AgeAtG1S'])[0] for c in collated])
Bsize = np.array([c['ActinSegmentationArea'].tolist()[0] for c in collated])
Region = np.array([np.unique(c['Region'])[0] for c in collated])
G1Growth = np.array([np.array(c['G1Growth'])[0] for c in collated])


Rg1 = np.zeros(Nregions); Rg1_censor = np.zeros(Nregions)
Rcycle = np.zeros(Nregions); Rcycle_censor = np.zeros(Nregions)
# Look at basic linear regression
for i in range(Nregions):
    # Plot as scatter plot
    I = Region == regionID[i]
    bsize = Bsize[I,...]
    tlength = Tcycle[I,...]
    tg1 = Tg1[I,...]
    ax = plt.subplot(2,2,i+1)
    # Overlay linear regression
    sb.regplot(bsize,tlength,y_jitter=False,
               scatter_kws={'alpha':0.2, 's':50})
    plt.xlim([0,1500]),plt.ylim([-0.5,4.5])
    plt.xlabel('Birth size (Crosssection area, px^2)')
    plt.ylabel('Cell cycle duration (days)')
    plt.title(''.join( ('Region ',str(regionID[i])) ))
    
    # Get pearson corr
    Rg1[i] = stats.pearsonr(bsize,tg1)[0]
    Rcycle[i] = stats.pearsonr(bsize,tlength)[0]
#    ax.text(6,2,''.join( ('R = ', '{:04.3f}'.format(Rg1[i]) ) ))

    ## Censor Birth size outliers (5% top/bottom)
    [mini,maxi] = stats.mstats.mquantiles(bsize,[.05,.95])
    filtered = (bsize > mini) & (bsize < maxi)
    ax = plt.subplot(2,2,i+1)
    sb.regplot(bsize[filtered],tlength[filtered],
               scatter_kws={'alpha':0.1, 's':50},color='r')
    plt.xlim([0,1500]); plt.ylim([0,6.5])
    plt.xlabel('Birth size (Crosssection area, px^2)')
    plt.ylabel('Cell cycle duration (days)')
    plt.title(''.join( ('Region ',str(regionID[i])) ))
    
    # Get pearson corr
    Rg1_censor[i] = stats.pearsonr(bsize[filtered],tg1[filtered])[0]
    Rcycle_censor[i] = stats.pearsonr(bsize[filtered],tlength[filtered])[0]
    
#    ax.text(8,2,''.join( ('R(censored) = ', '{:04.3f}'.format(Rg1[i]) ) ))
    

    
    
## Look for binned birth size
plt.figure()
for i in range(Nregions):
    plt.subplot(2,2,i+1)
    I = Region == regionID[i]
    bsize = Bsize[I,...]
    tlength =Tcycle[I,...]
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
#tlength = jitter(Tcycle,.1)
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


plt.figure()
# Plot amount of growth in G1 v birth size
for i in range(Nregions):
    # Plot as scatter plot
    I = Region == regionID[i]
    bsize = Bsize[I,...]
    g1growth = G1Growth[I,...]
    ax = plt.subplot(2,2,i+1)
    # Overlay linear regression
    sb.regplot(bsize,g1growth,y_jitter=False,
               scatter_kws={'alpha':0.2, 's':50})
    plt.xlim([0,1200]),plt.ylim([-500,1000])
    bin_edges = stats.mstats.mquantiles(bsize,[0,.2,.4,.6,.8,1])
    plot_bin_means(bsize,g1growth,bin_edges)
    plt.xlabel('Birth size (Crosssection area, px^2)')
    plt.ylabel('Amount grown in G1 (px^2)')
    plt.title(''.join( ('Region ',str(regionID[i])) ))
    
    
plt.figure()
# Plot amount of growth in G1 v birth size
for i in range(Nregions):
    # Plot as scatter plot
    I = Region == regionID[i]
    bsize = Bsize[I,...]
    g1growth = G1Growth[I,...]
    ax = plt.subplot(2,2,i+1)
    # Overlay linear regression
    sb.regplot(bsize,g1growth+bsize,y_jitter=False,
               scatter_kws={'alpha':0.2, 's':50})
    bin_edges = stats.mstats.mquantiles(bsize,[0,.2,.4,.6,.8,1])
    plot_bin_means(bsize,g1growth+bsize,bin_edges)
    plt.xlim([0,1200]),plt.ylim([-100,2000])
    plt.xlabel('Size at G1 (Crosssection area, px^2)')
    plt.ylabel('Amount grown in G1 (px^2)')
    plt.title(''.join( ('Region ',str(regionID[i])) ))
    
    
    
    
    
    
    
plot_bin_means(g1['Nuclear area'],g1['E2F total'],g2_bin_edges)





def plot_bin_means(X,Y,bin_edges,color='g'):
    """
    Plot the mean/std values of Y given bin_edges in X
    
    """
    
    which_bin = np.digitize(X,bin_edges)
    Nbins = len(bin_edges)-1
    means = np.zeros(Nbins)
    stds = np.zeros(Nbins)
    bin_centers = np.zeros(Nbins)
    for b in range(Nbins):
        y = Y[which_bin == b+1]
        bin_centers[b] = (bin_edges[b] + bin_edges[b+1]) / 2
        means[b] = y.mean()
        stds[b] = y.std() / np.sqrt(len(y))
    plt.errorbar(bin_centers,means,stds,ecolor=color)




    

