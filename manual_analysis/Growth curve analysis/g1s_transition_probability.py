#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:00:34 2019

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale 
from sklearn.cross_decomposition import PLSCanonical

# Delete S/G2 after first time point
g1only = []
for c in collated_filtered:
    c = c[c['Phase'] != 'Daughter G1'].copy()
    g1 = c[c['Phase'] == 'G1']
    g2 = c[c['Phase'] == 'SG2']
    if len(g2) > 0:
        g1 = g1.append(g2.iloc[0])
        g1only.append(g1)
dfc_g1 = pd.concat(g1only)

# Construct the G1S transition time series
dfc_g1['G1S_logistic'] = (dfc_g1['Phase'] != 'G1').astype(int)
dfc_g1 = dfc_g1.rename({'Volume (sm)':'vol_sm'},axis=1)
dfc_g1 = dfc_g1.rename({'Growth rate (sm)':'gr_sm'},axis=1)
# Get rid of NaNs
I = ~np.isnan(dfc_g1['gr_sm'])
dfc_g1 = dfc_g1.loc[I]


############### Plot G1S logistic as function of size ###############
model = smf.logit('G1S_logistic ~ vol_sm', data=dfc_g1).fit()
model.summary()
mdpoint = - model.params['Intercept'] / model.params['vol_sm']

print "Mid point is: ",  mdpoint
x = dfc_g1['vol_sm'].values
y = dfc_g1['G1S_logistic'].values
plt.scatter( x, jitter(y,0.1) )
sb.regplot(data = dfc_g1,x='vol_sm',y='G1S_logistic',logistic=True,scatter=False)
plt.ylabel('G1/S transition')
plt.vlines([mdpoint],0,1)
expitvals = expit( (x * model.params['vol_sm']) + model.params['Intercept'])
I = np.argsort(expitvals)
plt.plot(x[I],expitvals[I],'b')

# Plot ROC
y = dfc_g1['G1S_logistic']
x = dfc_g1['vol_sm']
x = sm.add_constant(x)
y_pred = model.predict(x)
fpr,tpr, thresholds = roc_curve(y,y_pred)
print 'Area under ROC: ', auc(fpr,tpr)
plt.figure(1)
plt.plot(fpr,tpr)
plt.xlim([0,1])


############### G1S logistic as function of age ###############
model = smf.logit('G1S_logistic ~ Age',data=dfc_g1).fit()
model.summary()
mdpoint = - model.params['Intercept'] / model.params['Age']

print "Mid point is: ",  mdpoint
x = dfc_g1['Age'].values
y = dfc_g1['G1S_logistic'].values
plt.scatter( x, jitter(y,0.1) )
sb.regplot(data = dfc_g1,x='Age',y='G1S_logistic',logistic=True,scatter=False)
plt.ylabel('G1/S transition')
plt.vlines([mdpoint],0,1)
expitvals = expit( (x * model.params['Age']) + model.params['Intercept'])
I = np.argsort(expitvals)
plt.plot(x[I],expitvals[I],'b')

# Plot ROC
y = dfc_g1['G1S_logistic']
x = dfc_g1['Age']
x = sm.add_constant(x)
y_pred = model.predict(x)
fpr,tpr, thresholds = roc_curve(y,y_pred)
print 'Area under ROC: ', auc(fpr,tpr)
plt.figure(1)
plt.plot(fpr,tpr)
plt.xlim([0,1])
plt.gca().set_aspect('equal', adjustable='box')




############### G1S logistic multiregression with: vol, gr, Age ###############
logit_model = smf.logit('G1S_logistic ~ vol_sm + gr_sm + Age', data = dfc_g1).fit()
logit_model.summary()
I = ~np.isnan(dfc_g1['gr_sm'])
y = dfc_g1.loc[I]['G1S_logistic']
y_pred = logit_model.predict()
fpr,tpr, thresholds = roc_curve(y,y_pred)
print 'Area under ROC: ', auc(fpr,tpr)
plt.figure(1)
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1])
plt.xlim([0,1])
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(['Cell volume','Age','Both'])




#NB: Strong colinearity between Age and Volume

# Transition rate prediction using PLS
X = dfc_g1[['vol_sm','Age','gr_sm']] # Design matrix
y = dfc_g1['G1S_logistic'] # Response var
# Drop NaN rows
I = np.isnan(dfc_g1['gr_sm'])
X = X.loc[~I].copy()
y = y[~I]
pls_model = PLSCanonical()
pls_model.fit(scale(X),y)

X_c,y_c = pls_model.transform(scale(X),y)





# Multiple linearregression on birth size and growth rate
df['bvol'] = df['Birth volume']
df['exp_gr'] = df['Exponential growth rate']
df['g1_len'] = df['G1 length']
model = smf.ols('g1_len ~ exp_gr + bvol', data = df).fit()
model.summary()
print model.pvalues




# Delete S/G2 after first time point
g1s_marked = []
for c in collated_filtered:
    c = c[c['Phase'] != 'Daughter G1'].copy()
    g1 = c[c['Phase'] == 'G1']
    g1['G1S_mark'] = 0
    g2 = c[c['Phase'] != 'G1'].reset_index()
    if len(g2) > 0:
        g2['G1S_mark'] = 0
        g2.at[0,'G1S_mark'] = 1

    g1 = g1.append(g2)
    g1s_marked.append(g1)
dfc_marked = pd.concat(g1s_marked)
dfc_marked = dfc_marked.drop(columns=['level_0','index'])

# Plot G1S transition rate as function of size
# Construct the G1S transition time series
dfc_marked = dfc_marked.rename({'Volume (sm)':'vol_sm'},axis=1)
dfc_marked = dfc_marked.rename({'Growth rate (sm)':'gr_sm'},axis=1)




# generate size bins
nbins = 10
vol_range = [dfc_marked['vol_sm'].min(),dfc_marked['vol_sm'].max()]
vol_bin_edges = np.linspace(vol_range[0],vol_range[1],nbins+1)
vol_bin_centers = np.array([ (vol_bin_edges[i]+vol_bin_edges[i+1])/2 for i in range(nbins) ])

count_g1s = np.empty(nbins) * np.nan
total_count = np.empty(nbins) * np.nan
for i in range(nbins):
    left_edge = vol_bin_edges[i]
    right_edge = vol_bin_edges[i+1]
    I = (dfc_marked['vol_sm'] > left_edge) & (dfc_marked['vol_sm'] <= right_edge)
    X = dfc_marked.loc[I]
    
    count_g1s[i] = np.float(X['G1S_mark'].sum())
    total_count[i] = len(X)
    
count_g1s[total_count < 4] = np.nan
total_count[total_count < 4] = np.nan

plt.plot(vol_bin_centers,count_g1s.astype(np.float)/total_count)
plt.xlabel('Cell volume (um3)')
plt.ylabel('G1/S transition rate')


