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
sb.regplot(data = dfc_no_daughters,x='Volume',y='G1S_logistic',logistic=True)
plt.ylabel('G1/S transition')
plt.vlines([mdpoint],0,1)


# Multiple linearregression on birth size and growth rate
df['bvol'] = df['Birth volume']
df['exp_gr'] = df['Exponential growth rate']
df['g1_len'] = df['G1 length']
model = smf.ols('g1_len ~ exp_gr + bvol', data = df).fit()
model.summary()

