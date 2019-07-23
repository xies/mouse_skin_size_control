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
from scipy import stats, optimize
import pickle as pkl

#Load df from pickle
r1 = pd.read_pickle('/Users/xies/Box/Mouse/Skin/W-R1/tracked_cells/dataframe.pkl')
r2 = pd.read_pickle('/Users/xies/Box/Mouse/Skin/W-R2/tracked_cells/dataframe.pkl')
r5 = pd.read_pickle('/Users/xies/Box/Mouse/Skin/W-R5/tracked_cells/dataframe.pkl')
df = pd.concat((r1,r2,r5))

# Load growth curves from pickle
with open('/Users/xies/Box/Mouse/Skin/W-R1/tracked_cells/collated_manual.pkl','rb') as f:
    c1 = pkl.load(f)
with open('/Users/xies/Box/Mouse/Skin/W-R2/tracked_cells/collated_manual.pkl','rb') as f:
    c2 = pkl.load(f)
with open('/Users/xies/Box/Mouse/Skin/W-R5/tracked_cells/collated_manual.pkl','rb') as f:
    c5 = pkl.load(f)
collated = c1 + c2 + c5

######

# p1 = multiplicative constant
# p2 = growth rate
# p3 = constant offset
exp_model = lambda x,p1,p2,p3 : p1 * np.exp(p2 * x) + p3

res = []
exp_b = []

# Fit Exponential model to growth curves
for c in collated:
    if c.iloc[0]['Phase'] == '?':
        continue
    c = c[ c['Daughter'] == 'None' ]
    if len(c) > 4:
#        t = np.arange(-g1sframe + 1,len(c) - g1sframe + 1) * 12 # In hours
        t = np.arange(len(c)) * 12
        v = c.Volume.values
        # Construct initial guess for growth rate

        # Nonlinear regression
        try:
            b = optimize.curve_fit(exp_model,t,v,p0 = [v[0],1,v.min()],
                                         bounds = [ [0,0,v.min()],
                                                    [v.max(),np.inf,v.max()]])
            exp_b.append(b)
            yhat = exp_model(t,b[0][0],b[0][1],b[0][2])
            res.append( v - yhat )
            plt.plot(t,v,'b')
            plt.plot(t,yhat,'g')

        except:
            print 'Fitting failed for ', c.iloc[0].CellID
            
all_res_exp = np.hstack(res)


# Fit linear model to growth curves
for c in collated:
    c = c[ c['Daughter'] == 'None' ]
    if c.iloc[0]['Phase'] == '?':
        continue
    if len(c) > 4:
#        t = np.arange(-g1sframe + 1,len(c) - g1sframe + 1) * 12 # In hours
        t = np.arange(len(c)) * 12
        v = c.Volume.values
        # Construct initial guess for growth rate

        # Linear regression
        p = np.polyfit(t,v,1)
        yhat = np.polyval(p,t)
        res.append( v - yhat )
        plt.plot(t,v,'b')
        plt.plot(t,yhat,'g')
        
        
        
all_res_lin = np.hstack(res)

plt.figure(1)
bins = np.linspace(-200,200,25)
plt.hist(all_res_exp,bins,histtype='step',density=True,stacked=True)
plt.xlabel('Fitting residuals (um3)')
plt.ylabel('Frequency')

plt.figure(1)
bins = np.linspace(-200,200,25)
N,bins,p = plt.hist(all_res_lin,bins,histtype='step',density=True,stacked=True)
plt.xlabel('Fitting residuals (um3)')
plt.ylabel('Frequency')


plt.figure(2)
bins = np.linspace(0,250,25)
plt.hist(np.abs(all_res_exp),bins,cumulative=True,normed=True,histtype='step')
plt.xlabel('Absolute residuals (um3)')
plt.ylabel('Frequency')

plt.figure(2)
bins = np.linspace(0,250,25)
plt.hist(np.abs(all_res_lin),bins,histtype='step',normed=True,cumulative=True)
plt.xlabel('Absolute residuals (um3)')
plt.ylabel('Frequency')

        
