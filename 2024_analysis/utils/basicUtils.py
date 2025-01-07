#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:11:53 2016

@author: mimi
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from SelectFromCollection import SelectFromCollection
from matplotlib.path import Path
from scipy import stats


def euclidean_distance(X,Y):
    X = np.array(X,dtype=float)
    Y = np.array(Y,dtype=float)
    assert(X.ndim == Y.ndim)
    return np.sqrt( np.sum((X-Y)**2) )


def plot_stack(im1,cmap1 = 'gray'):

    Z = im1.shape[0]
    nrows = np.int(np.ceil(np.sqrt(Z)))
    ncols = np.int(Z // nrows + 1)
        
    cmax = im1.max()
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    
    for z in range(Z):
        i = z // ncols
        j = z % ncols
        axes[i, j].imshow(im1[z, ...], cmap=cmap1, vmax=cmax)
        
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
    
    ## Remove empty plots 
    for ax in axes.ravel():
        if not(len(ax.images)):
            fig.delaxes(ax)
            
    fig.tight_layout()

def plot_2stacks(im1,im2,cmap1 = 'gray',cmap2='Dark2'):
    assert im1.shape == im2.shape,"Both images must be the same size!"

    Z = im1.shape[0]
    nrows = np.int(np.ceil(np.sqrt(Z)))
    ncols = np.int(Z // nrows + 1)
        
    fig, axes = plt.subplots(nrows, ncols*2, figsize=(3*ncols, 1.5*nrows))
    cmax1 = im1.max(); cmax2 = im2.max()
    
    cmin1 = im1.min(); cmin2 = im2.min()
    
    for z in range(Z):
        i = z // ncols
        j = z % ncols * 2
        axes[i, j].imshow(im1[z, ...], cmap=cmap1, vmax=cmax1, vmin=cmin1)
        axes[i, j+1].imshow(im2[z, ...], cmap=cmap2, vmax=cmax2, vmin=cmin2)
        
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        axes[i, j+1].set_xticks([])
        axes[i, j+1].set_yticks([])
    
    ## Remove empty plots 
    for ax in axes.ravel():
        if not(len(ax.images)):
            fig.delaxes(ax)
            
    fig.tight_layout()

def plot_stack_projections(image_stack,xy_scale,z_scale):
    fig = plt.figure(figsize=(12, 12))
    
    # xy projection:
    ax_xy = fig.add_subplot(111)
    ax_xy.imshow(image_stack.max(axis=0), cmap='gray')
    
    # ZX projection
    divider = make_axes_locatable(ax_xy)
    ax_zx = divider.append_axes("top", 2, pad=0.2, sharex=ax_xy)
    ax_zx.imshow(image_stack.max(axis=1), aspect=z_scale/xy_scale, cmap='gray')

    # YZ projection
    ax_yz = divider.append_axes("right", 2, pad=0.2, sharey=ax_xy)
    ax_yz.imshow(image_stack.max(axis=2).T, aspect=xy_scale/z_scale, cmap='gray')
    plt.draw()

#
#def sitk_show(img, title=None, margin=0.05, dpi=40 ):
#    nda = SimpleITK.GetArrayFromImage(img)
#    spacing = img.GetSpacing()
#    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
#    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
#    fig = plt.figure(figsize=figsize, dpi=dpi)
#    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
#
#    plt.set_cmap("gray")
#    ax.imshow(nda,extent=extent,interpolation=None)
#    
#    if title:
#        plt.title(title)
#    
#    plt.show()


def standardize(x):
    return (x / np.nanmean(x))

def z_standardize(x):
    return (x - np.nanmean(x))/np.nanstd(x)

def df_average(df, weights_column):
    '''Computes the average on each columns of a dataframe, weighted
    by the values of the column `weight_columns`.
    
    Parameters:
    -----------
    df: a pandas DataFrame instance
    weights_column: a string, the column name of the weights column 
    
    Returns:
    --------
    
    values: pandas DataFrame instance with the same column names as `df`
        with the weighted average value of the column
    '''
    
    values = df.copy().iloc[0]
    norm = df[weights_column].sum()
    for col in df.columns:
        try:
            v = (df[col] * df[weights_column]).sum() / norm
        except TypeError:
            v = df[col].iloc[0]
        values[col] = v
    return values


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError,"Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    if len(y) is not len(x):
        y = y[window_len/2-1:-(window_len/2)]
    return y

def plot_bin_means(X,Y,bin_edges=None,mean='median',error='sem',color=None,
                   style='errorbar',minimum_n=25,bin_style='equal'):
    """
    Plot the mean/std values of Y given bin_edges in X
    
    INPUT:
        X, Y - the X and Y of the datat to bin over
        bin_edges - edges of binning
        mean - 'mean' or 'median'
        error - 'sem' (default) for standard error of mean or 'std' for standard deviation
        color - color to pass to errorbar
        minimum_n - minimum # of points per bin (default = 25)
        bin_style - 'equal' for equal spacing; 'percentile' for cohort
    
    RETURN:
        mean,std
    """
    
    assert(X.shape == Y.shape)
    
    X,Y = nonan_pairs(X,Y)
    
    # Flatten if not vectors
    if X.ndim > 1:
        X = X.flatten()
        Y = Y.flatten()
    
    if type(bin_edges) == int:
        if bin_style == 'equal':
            X_min = X.min()
            X_max = X.max() 
            bin_edges = np.linspace(X_min,X_max,num=bin_edges)
        elif bin_style == 'percentile':
            bin_edges = np.percentile(nonans(X),np.linspace(0,100,num=bin_edges))
            print(bin_edges)
    else:
        raise ValueError
    print(X)
    which_bin = np.digitize(X,bin_edges)
    Nbins = len(bin_edges)-1
    means = np.zeros(Nbins)
    stds = np.zeros(Nbins)
    
    
    # bin_centers = np.zeros(Nbins)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    
    for b in range(Nbins):
        y = Y[which_bin == b+1]
        # bin_centers[b] = (bin_edges[b] + bin_edges[b+1]) / 2
        # Suppress noisy bins
        if len(y) < minimum_n:
            means[b] = np.nan
            stds[b] = np.nan
        else:
            # Mean or median
            if mean == 'mean':
                means[b] = np.nanmean(y)
            elif mean == 'median':
                print(f'{y.shape}')
                means[b] = np.nanmedian(y)
    
            if error == 'sem':
                stds[b] = np.nanstd(y) / np.sqrt(len(y))
            elif error == 'std':
                stds[b] = y.std()

    # Plot
    if style == 'errorbar':
        plt.errorbar(bin_centers,means,stds,color=color)
    elif style == 'fill':
        plt.plot(bin_centers, means, color=color)
        plt.fill_between(bin_centers, means-stds, means+stds,
                         color=color,alpha=0.5)
        
    return means

def get_bin_means(X,Y,bin_edges=None,mean='median',error='sem',minimum_n=25):
    """
    Get the mean/std values of Y given bin_edges in X
    
    INPUT:
        X, Y - the X and Y of the datat to bin over
        bin_edges - edges of binning
        mean - 'mean' or 'median'
        error - 'sem' (default) for standard error of mean or 'std' for standard deviation
        color - color to pass to errorbar
        minimum_n - minimum # of points per bin (default = 10)
    
    RETURN:
        mean,std
    """
    
    assert(X.shape == Y.shape)
    
    # Flatten if not vectors
    if X.ndim > 1:
        X = X.flatten()
        Y = Y.flatten()
    
    if (bin_edges == None).all():
        X_min = np.nanmin(X)
        X_max = np.nanmax(X)
        bin_edges = np.linspace(X_min,X_max,num=10)
    
    
    which_bin = np.digitize(X,bin_edges)
    Nbins = len(bin_edges)-1
    # Calculate x-errors
    xmeans = np.array([X[ which_bin == i ].mean() for i in range(which_bin.max())])
    xerrors = np.array([X[ which_bin == i ].std() for i in range(which_bin.max())])
    
    means = np.zeros(Nbins)
    stds = np.zeros(Nbins)
    bin_centers = np.zeros(Nbins)
    for b in range(Nbins):
        y = Y[which_bin == b+1]
        bin_centers[b] = (bin_edges[b] + bin_edges[b+1]) / 2
        # Suppress noisy bins
        if len(y) < minimum_n:
            means[b] = np.nan
            stds[b] = np.nan
        else:
            # Mean or median
            if mean == 'mean':
                means[b] = np.nanmean(y)
            elif mean == 'median':
                means[b] = np.nanmedian(y)
    
            if error == 'sem':
                stds[b] = np.nanstd(y) / np.sqrt(len(y))
            elif error == 'std':
                stds[b] = np.nanstd(y)
        
    return means,stds,xmeans,xerrors


def plot_slopegraph(X,Y,color='b',names=None):
    """
    Implements a Tufte's slopegraph for two paired lists
    
    Inputs:
        X,Y paired
        color - optional, default 'b'
        names - [X_name,Y_name] (optional)
    """
    
    assert(len(X) == len(Y)), 'X and Y must have same length'
    assert( (np.ndim(X) == np.ndim(Y)) & np.ndim(X) == 1 ), 'X and Y must be 1-dimensional arrays'
    
    N = len(X)
    for i in range(N):
        # Skip if one of value is NaN
        x = X[i]; y = Y[i]
        if ~np.isnan(x) and ~np.isnan(y):
            # Plot X, Y as scatter first
            plt.scatter([1,2],[x,y],color=color)
            # Plot slope
            plt.plot([1,2],[x,y],color=color)
            if names is not None:
                plt.xticks([1,2],names)

def standardize_df(df,excepting=None):
    fields = df.columns
    df_new = df.copy()
    

def nan_polyfit(x,y,deg):
    # Gets rid of nans and pass to numpy polyfit
    I = ~np.isnan(x)
    I = I & ~(np.isnan(y))
    p,R = np.polyfit(x[I], y[I], deg)
    return p,R

def overlap(a, b):
    return min(a[1],b[1]) - max(a[0],b[0])

def nonans(x):
    return x[~np.isnan(x)]

def nonan_pairs(x,y):
    I = ~np.isnan(x)
    I = I & ~(np.isnan(y))
    return x[I],y[I]

def jitter(x,sigma):
    N = len(x)
    noise = random.rand(N)
    return x + (noise - 0.5) * sigma


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


def find_last_n_nonnans(x,n):
    '''
    Return the last n non-NaN elements of the array x
    '''
    I = np.sort( np.where( ~np.isnan(x) ) )[0]
    
    return I[-1-n:-1]

def find_nearest_idx(array,value):
    '''
    Find the index of the element in an array nearest to a given search value
    '''
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx
    

def draw_gate(df,x,y,xlims=None,ylims=None,alpha=0.1):
    
    
    plt.figure()
    pts = plt.scatter(df[x],df[y],alpha=alpha)

    if xlims:
        plt.xlim(xlims)
    if ylims:
        plt.ylim(ylims)

    selector = SelectFromCollection(plt.gca(), pts)
    return selector

def gate_on_selector(selector,df,xname,yname):
    verts = np.array(selector.poly.verts)
    x = verts[:,0]
    y = verts[:,1]
    
    p_ = Path(np.array([x,y]).T)
    I = np.array([p_.contains_point([x,y]) for x,y in zip(df[xname],df[yname])])

    return I

def ttest_from_groupby(df,field2group,field2test):
    
    grouped = list(df.groupby(field2group))
    assert(len(grouped) == 2)
    
    X = grouped[0][1]
    Y = grouped[1][1]
    
    X = X[field2test].dropna()
    Y = Y[field2test].dropna()
    
    T,P = stats.ttest_ind(X,Y)
    
    return T,P


def kstest_from_groupby(df,field2group,field2test):
    
    grouped = list(df.groupby(field2group))
    assert(len(grouped) == 2)
    
    X = grouped[0][1]
    Y = grouped[1][1]
    
    X = X[field2test].dropna()
    Y = Y[field2test].dropna()
    
    T,P = stats.kstest(X,Y)
    
    return T,P

def plot_from_groupby(df,field2group,field2plot,plotfunc):
    grouped = list(df.groupby(field2group))
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    legends = []
    for i,(name,df_) in enumerate(grouped):
        X = df_[field2plot]
        plotfunc(X, color=colors[i])
        legends.append(name)

    plt.legend(legends)
    return
    

def normplot(x,**kwargs):
    
    x = nonans(x)
    (qtile_counts,qtile_bins), (slope, intercept, r) = stats.probplot(x,dist='norm')
    plt.scatter(qtile_bins,qtile_counts, **kwargs)
    
    plt.plot(qtile_counts * slope + intercept, qtile_counts,'k--',label='_nolegend_') # no legend
    
    # ticks
    ticks_perc=[1, 5, 10, 20, 50, 80, 90, 95, 99]
    ticks_quan=[stats.norm.ppf(i/100.) for i in ticks_perc]

    #assign new ticks
    plt.yticks(ticks_quan,ticks_perc)
    plt.xlabel('Values')
    plt.ylabel('Quantiles')
    
    plt.grid()
    plt.show()
    
    
    return



