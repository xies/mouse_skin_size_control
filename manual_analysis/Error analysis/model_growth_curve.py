#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 19:06:55 2019

@author: xies
"""

import seaborn as sb
from scipy import optimize,stats
from scipy.interpolate import UnivariateSpline

###### Models + spline fit

sb.set_style('darkgrid')

# p1 = multiplicative constant
# p2 = growth rate
# p3 = constant offset
exp_model = lambda x,p1,p2,p3 : p1 * np.exp(p2 * x) + p3

res_lin = []
res_exp = []
res_spl = []
yhat_spl = []
phases = []

# Properly store the exponential growth rate (p2)
regionCellIDs = (df['Region'] + df['CellID'].astype(int).astype(str)).values.tolist()
exp_b = np.empty(Ncells) * np.nan

#counter = 0
# Fit Exponential & linear models to growth curves
for c in collated_filtered:
    if c.iloc[0]['Phase'] == '?':
        continue
    c = c[ c['Daughter'] == 'None' ]
    if len(c) > 4:
#        t = np.arange(-g1sframe + 1,len(c) - g1sframe + 1) * 12 # In hours
        t = np.arange(len(c)) * 12
        v = c.Volume.values
        # Construct initial guess for growth rate

#        try:
            # Nonlinear regression
#            b = optimize.curve_fit(exp_model,t,v,p0 = [v[0],1,v.min()],
#                                         bounds = [ [0,0,v.min()],
#                                                    [v.max(),np.inf,v.max()]])
#            yhat = exp_model(t,b[0][0],b[0][1],b[0][2])
#            res_exp.append( (v - yhat)/v )
#            idx = regionCellIDs.index(c.iloc[0]['Region']+c.iloc[0]['CellID'].astype(str))
#            exp_b[idx] = b[0][1]
            
    
    #            plt.subplot(2,3,counter+1)
#        plt.plot(t,v,'k')
#        plt.plot(t,yhat,'g')
        
#            # LInear regression
#            p = np.polyfit(t,v,1)
#            yhat = np.polyval(p,t)
#            res_lin.append( v - yhat )
#            plt.plot(t,yhat,'r')
        
        # B-spline
        spl = UnivariateSpline(t, v, k=3, s=1e6)
        plt.plot(t,spl(t),'b')
        yhat_spl.append(spl(t))
        res_spl.append( (v-spl(t))/v )
        phases.append(c['Phase'])
        
        plt.xlabel('Time since birth (hr)')
        plt.ylabel('Cell volume')
#            plt.legend(('Data','Exponential model','Linear model','Cubic spline'))
            
#            counter += 1
#            if counter > 5:
#                break
            
        except:
            print 'Fitting failed for ', c.iloc[0].CellID
            

all_res_exp = np.hstack(res_exp)
all_res_lin = np.hstack(res_lin)
all_res_spl = np.hstack(res_spl)

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

plt.figure(1)
bins = np.linspace(-200,200,25)
N,bins,p = plt.hist(all_res_spl,bins,histtype='step',density=True,stacked=True)
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

plt.figure(2)
bins = np.linspace(0,250,25)
plt.hist(np.abs(all_res_spl),bins,histtype='step',normed=True,cumulative=True)
plt.xlabel('Absolute residuals (um3)')
plt.ylabel('Frequency')
=


