#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:00:34 2019

@author: xies
"""

import numpy as np
import pandas as pd
from numpy import random
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit
from sklearn.cross_decomposition import PLSRegression, PLSSVD

# Delete S/G2 after first time point
g1only = []
for c in collated_filtered:
    c = c[c['Phase'] != 'Daughter G1'].copy()
    g1 = c[c['Phase'] == 'G1']
    g2 = c[c['Phase'] == 'SG2']
    if len(g2) > 0:
        g1 = g1.append(g2.iloc[0])
        g1only.append(g1)
dfc = pd.concat(g1only)

# Construct the G1S transition time series
dfc['G1S_logistic'] = (dfc['Phase'] != 'G1').astype(int)
dfc_no_daughters = dfc[dfc['Phase'] != 'Daughter G1']

# Plot G1S trans as function of size
y = dfc_no_daughters['G1S_logistic']
x = dfc_no_daughters['Volume']
x = sm.add_constant(x)
model = sm.Logit(y,x).fit()
model.summary()
mdpoint = - model.params['const'] / model.params['Volume']

print "Mid point is: ",  mdpoint
x = dfc_no_daughters['Volume'].values
y = dfc_no_daughters['G1S_logistic'].values
plt.scatter( x, jitter(y,0.1) )
sb.regplot(data = dfc_no_daughters,x='Volume',y='G1S_logistic',logistic=True,scatter=False)
plt.ylabel('G1/S transition')
plt.vlines([mdpoint],0,1)
#expitvals = special.expit( (x * model.params['Volume']) + model.params['const'])
#I = np.argsort(expitvals)
#plt.plot(x[I],expitvals[I],'b')




# Transition rate prediction
dfc = dfc.rename({'Volume (sm)':'vol_sm'},axis=1)
dfc = dfc.rename({'Growth rate (sm)':'gr_sm'},axis=1)

model = smf.logit('G1S_logistic ~ vol_sm + gr_sm + Age', data = dfc).fit()
model.summary()

#NB: Strong colinearity between Age and Volume

# Transition rate prediction using PLS




# Multiple linearregression on birth size and growth rate
df['bvol'] = df['Birth volume']
df['exp_gr'] = df['Exponential growth rate']
df['g1_len'] = df['G1 length']
model = smf.ols('g1_len ~ exp_gr + bvol', data = df).fit()
model.summary()
print model.pvalues





