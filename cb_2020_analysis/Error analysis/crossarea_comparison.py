#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 19:06:55 2019

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pickle as pkl
from scipy import optimize, stats
from scipy.interpolate import UnivariateSpline

# Load growth curves from pickle
with open('/Users/xies/Box/Mouse/Skin/W-R1/tracked_cells/collated_manual.pkl','rb') as f:
    c1 = pkl.load(f)
with open('/Users/xies/Box/Mouse/Skin/W-R2/tracked_cells/collated_manual.pkl','rb') as f:
    c2 = pkl.load(f)
with open('/Users/xies/Box/Mouse/Skin/W-R5/tracked_cells/collated_manual.pkl','rb') as f:
    c5 = pkl.load(f)
collated = c1 + c2 + c5

dx = 0.25

#######################################
# Grab the automatic trancked data
dirname = '/Users/xies/Box/Mouse/Skin/W-R1/'
with open(path.join(dirname,'collated.pkl'),'rb') as f:
    a1 = pkl.load(f)

dirname = '/Users/xies/Box/Mouse/Skin/W-R2/'
with open(path.join(dirname,'collated.pkl'),'rb') as f:
    a2 = pkl.load(f)

dirname = '/Users/xies/Box/Mouse/Skin/W-R5/'
with open(path.join(dirname,'collated.pkl'),'rb') as f:
    a5 = pkl.load(f)
auto_tracked = a1+a2+a5
autoIDs = np.array([c.CellID.iloc[0] for c in auto_tracked])

#####
areas = []
volumes = []
phases = []
nuc = []
# Generate scatter plot of area v. volume
aIDs = [(a.iloc[0].CellID,a.iloc[0].Region) for a in auto_tracked]
for c in collated_filtered:
    c = c[c['Daughter'] == 'None']
    # Find the automatic cell corresponding to it
    thisID = (c.iloc[0].CellID,c.iloc[0].Region)
    if thisID in aIDs:
        a = auto_tracked[aIDs.index(thisID)]
        if len(c) == len(a):
            areas += (a.ActinSegmentationArea.values * dx**2 ).tolist()
            volumes += c.Volume.values.tolist()
            phases += c.Phase.values.tolist()
            nuc += c.Nucleus.values.tolist()
plt.scatter(volumes,areas)
plt.xlabel('Cell volume (um3)')
plt.ylabel('Cross sectional area (um2)')

# Calculate R^2

slope, intercept, r_value, p_value, std_err = stats.linregress(volumes, areas)

###### Models + spline fit

sb.set_style('darkgrid')

# p1 = multiplicative constant
# p2 = growth rate
# p3 = constant offset
exp_model = lambda x,p1,p2,p3 : p1 * np.exp(p2 * x) + p3

#yhat_spl = []
res_exp = []
res_spl = []
nknots = []

# Fit Exponential & linear models to growth curves
# Quantify residuals
#vcellIDs = np.array([c.iloc[0].CellID for c in a2])
#indices = np.where( np.in1d(vcellIDs,[380,376,639,602,730]) )[0]

for c in auto_tracked:
    if len(c) > 3:
#        t = np.arange(-g1sframe + 1,len(c) - g1sframe + 1) * 12 # In hours
        t = np.arange(len(c)) * 12
        v = c.ActinSegmentationArea.values * dx**2
        
        try:
            # Nonlinear regression exponential
            b = optimize.curve_fit(exp_model,t,v,p0 = [v[0],1,v.min()],
                                         bounds = [ [0,0,v.min()],
                                                    [v.max(),np.inf,v.max()]])
            yhat = exp_model(t,b[0][0],b[0][1],b[0][2])
            res_exp.append( (v - yhat)/v )
            
            # B-spline
            spl = UnivariateSpline(t, v, k=3, s=1e6)
#            yhat_spl.append(spl(t))
            res_spl.append( (v - spl(t)) /v)
            nknots.append(len(spl.get_knots()))
            
#            plt.subplot(2,3,counter+1)
#            plt.plot(t,spl(t),'g')
#            plt.plot(t,v,'b')
            
        except:
            print 'Fitting failed for ', c.iloc[0].CellID
            
#auto_res_exp = np.hstack(res_exp)
auto_res_spl = np.hstack(res_spl)
auto_res_exp = np.hstack(res_exp)

plt.figure(1)
weights = np.ones_like(auto_res_exp)/float(len(auto_res_exp))
bins = np.linspace(-1,1,25)
plt.hist(auto_res_exp,bins,histtype='step',density=False,stacked=True,weights=weights)
plt.xlabel('Fitting residuals (um3)')
plt.ylabel('Frequency')

plt.figure(1)
weights = np.ones_like(auto_res_spl)/float(len(auto_res_spl))
bins = np.linspace(-1,1,25)
N,bins,p = plt.hist(auto_res_spl,bins,histtype='step',density=False,stacked=True,weights=weights)
plt.xlabel('Normalized residuals (um3)')
plt.ylabel('Frequency')

plt.legend(('Area','Volume'))
