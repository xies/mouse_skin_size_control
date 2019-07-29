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
from scipy import optimize
from scipy.interpolate import UnivariateSpline

# Load growth curves from pickle
with open('/Users/xies/Box/Mouse/Skin/W-R1/tracked_cells/collated_manual.pkl','rb') as f:
    c1 = pkl.load(f)
with open('/Users/xies/Box/Mouse/Skin/W-R2/tracked_cells/collated_manual.pkl','rb') as f:
    c2 = pkl.load(f)
with open('/Users/xies/Box/Mouse/Skin/W-R5/tracked_cells/collated_manual.pkl','rb') as f:
    c5 = pkl.load(f)
collated = c1 + c2 + c5

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
# Grab the automatic trancked data and look at how they relate
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
auto = []

for i in range(Ncells):
    ind = np.where(autoIDs == collated[i].CellID.iloc[0])[0][0]
    auto.append(auto_tracked[ind])


###### Models + spline fit

sb.set_style('darkgrid')

# p1 = multiplicative constant
# p2 = growth rate
# p3 = constant offset
exp_model = lambda x,p1,p2,p3 : p1 * np.exp(p2 * x) + p3

res_lin = []
res_exp = []
res_spl = []
exp_b = []
yhat_spl = []

#counter = 0
# Fit Exponential & linear models to growth curves
for c in auto_tracked:
    if len(c) > 4:
#        t = np.arange(-g1sframe + 1,len(c) - g1sframe + 1) * 12 # In hours
        t = np.arange(len(c)) * 12
        v = c.ActinSegmentationArea.values * dx**2
        # Construct initial guess for growth rate

        try:
            # Nonlinear regression
            b = optimize.curve_fit(exp_model,t,v,p0 = [v[0],1,v.min()],
                                         bounds = [ [0,0,v.min()],
                                                    [v.max(),np.inf,v.max()]])
            exp_b.append(b)
            yhat = exp_model(t,b[0][0],b[0][1],b[0][2])
            res_exp.append( (v - yhat)/v )
    
    #            plt.subplot(2,3,counter+1)
            plt.plot(t,v,'k')
            plt.plot(t,yhat,'g')
            
            # LInear regression
            p = np.polyfit(t,v,1)
            yhat = np.polyval(p,t)
            res_lin.append( v - yhat )
            plt.plot(t,yhat,'r')
            
            # B-spline
            spl = UnivariateSpline(t, v, k=3, s=1e6)
            plt.plot(t,spl(t),'b')
            yhat_spl.append(spl(t))
            
            plt.xlabel('Time since birth (hr)')
            plt.ylabel('Cell volume')
    #            plt.legend(('Data','Exponential model','Linear model','Cubic spline'))
                
#            counter += 1
#            if counter > 5:
#                break
            
        except:
            print 'Fitting failed for ', c.iloc[0].CellID
            

auto_res_exp = np.hstack(res_exp)
auto_res_lin = np.hstack(res_lin)
auto_res_spl = np.hstack(res_spl)

#plt.figure(1)
#bins = np.linspace(-200,200,25)
#plt.hist(all_res_exp,bins,histtype='step',density=True,stacked=True)
#plt.xlabel('Fitting residuals (um3)')
#plt.ylabel('Frequency')


plt.figure(1)
bins = np.linspace(-1,1,25)
plt.hist(auto_res_exp,bins,histtype='step',density=True,stacked=True)
plt.xlabel('Fitting residuals (um3)')
plt.ylabel('Frequency')

plt.figure(1)
bins = np.linspace(-1,1,25)
N,bins,p = plt.hist(all_res_exp,bins,histtype='step',density=True,stacked=True)
plt.xlabel('Normalized residuals (um3)')
plt.ylabel('Frequency')

plt.legend(('Area','Volume'))

#plt.figure(1)
#bins = np.linspace(-200,200,25)
#N,bins,p = plt.hist(all_res_spl,bins,histtype='step',density=True,stacked=True)
#plt.xlabel('Fitting residuals (um3)')
#plt.ylabel('Frequency')
