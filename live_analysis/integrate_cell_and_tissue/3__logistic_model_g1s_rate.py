#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:27:52 2022

@author: xies
"""


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit
from basicUtils import *

from numpy import random
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale 
from sklearn.cross_decomposition import PLSCanonical

def plot_logit_model(model,field):
    mdpoint = - model.params['Intercept'] / model.params[field]

    print(f'Mid point is: {mdpoint}')
    x = df_g1s[field].values
    y = df_g1s['G1S_logistic'].values
    plt.figure()
    plt.scatter( x, jitter(y,0.1) )
    sb.regplot(data = df_g1s,x=field,y='G1S_logistic',logistic=True,scatter=False)
    plt.ylabel('G1/S transition')
    plt.vlines([mdpoint],0,1)
    expitvals = expit( (x * model.params[field]) + model.params['Intercept'])
    I = np.argsort(expitvals)
    plt.plot(x[I],expitvals[I],'b')

#%% Data sanitization

df_ = df[df['Phase'] != '?']
df_g1s = pd.DataFrame()
df_g1s = df_.rename(columns={'Volume (sm)':'vol_sm'
                             ,'Coronal density':'cor_density'})
df_g1s['G1S_logistic'] = (df_['Phase'] == 'SG2').astype(int)

#%%

field = 'vol_sm'
############### Plot G1S logistic as function of size ###############
model = smf.logit(f'G1S_logistic ~ {field}', data=df_g1s).fit()
model.summary()
# plot_logit_model(model,field)

#%%

field = 'Age'
############### G1S logistic as function of age ###############
model = smf.logit(f'G1S_logistic ~ {field}',data=df_g1s).fit()
model.summary()
# plot_logit_model(model,field)

#%%

############### G1S logistic as function of age ###############
model = smf.logit(f'G1S_logistic ~ vol_sm + Age + cor_density',data=df_g1s).fit()
model.summary()


#%%

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


Ncells = len(dfc_g1)
Nboot = 1000
mp_bs = np.zeros(Nboot)
for i in range(Nboot):
#    df_bs = pd.DataFrame(dfc_g1.values[random.randint(Ncells, size=Ncells)], columns=dfc_g1.columns)
    df_bs = dfc_g1.iloc[random.randint(Ncells,size=Ncells)]
    m = smf.logit('G1S_logistic ~ Age', data=df_bs).fit()
    mp_bs[i] = - m.params['Intercept'] / m.params['Age']

sb.regplot(data = dfc_g1,x='Age',y='G1S_logistic',logistic=True,scatter=True)


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


