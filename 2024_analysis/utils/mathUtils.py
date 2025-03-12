#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 20:22:07 2022

@author: xies
"""

import numpy as np
from scipy import stats, linalg, optimize
from skimage import morphology
from basicUtils import nonans, nonan_pairs
from numpy import random
# from scipy import stats
import pandas as pd
from scipy.interpolate import UnivariateSpline

def exponential_growth(t,V0,k):
    return V0*np.exp(t*k)

def fit_exponential_curve(t,v):
    t,v = nonan_pairs(t,v)
    print(v)
    p0 = [v[0],v[1]/v[0]]
    params = optimize.curve_fit(exponential_growth,t,v,p0)
    return params

def get_interpolated_curve(cf,y_field='Volume',x_field='Age',smoothing_factor=1e10):

    # Get rid of daughter cells]
    if len(cf) < 4:
        yhat = cf[y_field].values
        dydt = np.ones(len(cf)) * np.nan
        
    else:
        t = cf[x_field].values
        v = cf[y_field].values
        # Spline smooth
        spl = UnivariateSpline(t, v, k=3, s=smoothing_factor)
        yhat = spl(t)
        
        dydt = spl.derivative(n=1)(t)
        
        # # Nuclear volume
        # nv = cf.Nucleus.values
        # # Spline smooth
        # spl = UnivariateSpline(t, nv, k=3, s=smoothing_factor)
        # nuc_hat = spl(t)

    return yhat,dydt


def total_std(means,stds,num_per_sample):
    assert(len(stds) == len(means))
    assert(len(stds) == len(num_per_sample))

    means = np.array(means)
    stds = np.array(stds)
    num_per_sample = np.array(num_per_sample)

    total_mean = (num_per_sample * means).sum() / num_per_sample.sum()
    D = means - total_mean

    total_std = ((num_per_sample * stds**2).sum() + (num_per_sample*D**2).sum()) / num_per_sample.sum()
    return np.sqrt(total_std)


def surface_area(im):
    assert(im.ndim == 3)
    eroded = morphology.binary_erosion(im).astype(int)
    outline = im - eroded
    return outline.sum()

def parse_3D_inertial_tensor(I):
    '''
    Takes a 3D inertial tensor and returns how aligned the major component is to Z-axis,
    the moments in order, and the planar orientation
    '''
    assert(I.shape == (3,3))
    L,D = linalg.eig(I)
    L = np.real(L) # Assume no complex solution is necessary

    #Sort eigvec by eigenvalue magnitudes
    order = np.flip( np.argsort(L) )
    L = L[order]
    D = D[:,order]

    Znorm = np.array([1,0,0]).T
    dots = np.array([ np.dot(d,Znorm) for d in D.T ])

    phis = np.rad2deg( np.arccos(np.abs(dots)) )

    # # Znorm = np.zeros((3,3))
    # # Znorm[0,:] = 1

    # # norms = np.sqrt( (delta**2).sum(axis=0) )

    # # Flip Znorm if everything is too large
    # if np.all(norms > 1):
    #     delta = D + Znorm
    #     norms = np.sqrt( (delta**2).sum(axis=0) )

    axis_order = np.argsort( phis )

    phi = phis[axis_order[0]]
    axial_moment = L[axis_order[0]]
    other_moments = L[axis_order[1:]]
    Z_proj_planar_major = D[1,1:] #NB: not unit!
    theta = np.arccos(Z_proj_planar_major[1] / linalg.norm(Z_proj_planar_major))
    if theta < 0:
        theta = theta + 180

    theta = np.rad2deg(theta)

    return axial_moment, phi, other_moments[0], other_moments[1], theta

def cvariation_ci(x,correction=True,alpha=0.05):
    '''Calculate the confidence interval of CV estimate. Omits NaNs

    Input:
        x: vector to calculate CV and CI on
        correction: default true, whether to do standard correction (Vangel method)
        alpha: test threshold (default = 0.05)

    Output:
        cv, lci,uci - Confidence interval at significance threshold requested

    Source: https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/coefvacl.htm

    '''

    # NaN-filter
    x = nonans(x)
    N = len(x)

    CV = np.std(x) / np.mean(x)

    # Get chi-sq stats
    u1 = stats.chi2.ppf(1-alpha/2,N-1)
    u2 = stats.chi2.ppf(alpha/2,N-1)
    # Corrected method
    if correction:
        LCI = CV / np.sqrt( ((u1+2)/N - 1)*CV**2 + u1 / (N-1) )
        UCI = CV / np.sqrt( ((u2+2)/N - 1)*CV**2 + u2 / (N-1) )
    else:
        # Uncorrected CI
        LCI = CV * np.sqrt((N-1)/u1)
        UCI = CV * np.sqrt((N-1)/u2)

    return [LCI,UCI]


def cvariation_bootstrap(x,Nboot,alpha=0.05,subsample=None):
    '''
    Calculates the confidence intervals of the CV of a sample using bootstrapping.
    Ignores NaNs
    '''
    x = nonans(x)
    if subsample is None:
        subsample = len(x)
    if isinstance(x,pd.core.series.Series):
        x = x.values
    _CV = [stats.variation(x[random.randint(low=0,high=len(x),size=subsample)]) for i in range(Nboot)]
    _CV = np.array(_CV)
    lb,ub = stats.mstats.mquantiles(_CV,prob = [alpha,1-alpha])
    return [np.nanmean(_CV),lb,ub]

def cv_difference_pvalue(x,y,Nboot, subsample=None):
    if subsample is None:
        subsample = min(len(x),len(y))
    if isinstance(x,pd.Series):
        x = x.values
    if isinstance(y,pd.Series):
        y = y.values
    
    _CVx = np.array([stats.variation(x[random.randint(low=0,high=len(x),size=subsample)]) for i in range(Nboot)])
    _CVy = np.array([stats.variation(y[random.randint(low=0,high=len(y),size=subsample)]) for i in range(Nboot)])
    
    P = (_CVx < _CVy).sum() / Nboot
    return P

# Construct triangulation
def adjmat2triangle(G):
    triangles = set()
    for u,w in G.edges:
        for v in set(G.neighbors(u)).intersection(G.neighbors(w)):
            triangles.add(frozenset([u,v,w]))
    return triangles


def get_neighbor_idx(tri,idx):
    neighbors = np.unique(tri.simplices[np.any(tri.simplices == idx,axis=1),:])
    neighbors = neighbors[neighbors != idx] # return only nonself
    return neighbors.astype(int)

def get_neighbor_distances(tri,idx,coords):
    neighbor_idx = get_neighbor_idx(tri,idx)
    this_coord = coords[idx,:]
    neighbor_coords = coords[neighbor_idx,:]
    D = np.array([euclidean_distance(this_coord,n) for n in neighbor_coords])
    return D

def argsort_counter_clockwise(X,Y):
    cx = X.mean()
    cy = Y.mean()

    dX = X - cx
    dY = Y - cy

    thetas = np.arctan2(dX,dY)
    return np.argsort(thetas)

def polygon_area(X,Y):

    S1 = np.sum(X*np.roll(Y,-1))
    S2 = np.sum(Y*np.roll(X,-1))
    area = .5*np.absolute(S1 - S2)

    return area

########################################################################################
# Author: Ujash Joshi, University of Toronto, 2017                                     #
# Based on Octave implementation by: Benjamin Eltzner, 2014 <b.eltzner@gmx.de>         #
# Octave/Matlab normxcorr2 implementation in python 3.5                                #
# Details:                                                                             #
# Normalized cross-correlation. Similiar results upto 3 significant digits.            #
# https://github.com/Sabrewarrior/normxcorr2-python/master/norxcorr2.py                #
# http://lordsabre.blogspot.ca/2017/09/matlab-normxcorr2-implemented-in-python.html    #
########################################################################################

from skimage.filters import gaussian
from scipy.signal import fftconvolve

def normxcorr2(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return np.abs(out)

def estimate_log_normal_parameters(sample_mu,sample_sigma):
    '''
    Convert a sample mean and std from a log-normal distr. and convert to the underlying
    normal distr mu and sigma

    Parameters
    ----------
    sample_mu : TYPE
        DESCRIPTION.
    sample_sigma : TYPE
        DESCRIPTION.

    Returns
    -------
    mu, sigma of 'not-log' normal distribution

    See: https://en.wikipedia.org/wiki/Log-normal_distribution

    '''
    mu = np.log(sample_mu**2 / np.sqrt(sample_mu**2+sample_sigma**2))
    sigma = np.sqrt( np.log(1+sample_sigma**2/sample_mu**2) )
    return mu,sigma
