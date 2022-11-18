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
from os import path

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

def z_standardize(x):
    return (x - np.nanmean(x))/np.std(x)

#%% Load features from training + test set

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
df = pd.read_csv(path.join(dirname,'MLR model/ts_features.csv'),index_col=0)
df_ = df[df['Phase'] != '?']

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'
df_test = pd.read_csv(path.join(dirname,'MLR model/ts_features.csv'),index_col=0)
df_test_ = df_test[df_test['Phase'] != '?']

# df_ = pd.concat((df_,df_test_),ignore_index=True)
N,P = df_.shape

#%% Sanitize field names for smf

features_list = { # Cell geometry
                'Age':'age'
                # ,'Z_x':'z','Y_x':'y','X_x':'x'
                ,'Volume (sm)':'vol_sm'
                ,'Axial component':'axial_moment'
                ,'Nuclear volume':'nuc_vol'
                # ,'Nuclear surface area':'nuc_sa'
                # ,'Nuclear axial component':'nuc_axial_moment'
                # ,'Nuclear solidity':'nuc_solid'
                ,'Nuclear axial angle':'nuc_angle'
                ,'Planar eccentricity':'planar_ecc'
                ,'Axial eccentricity':'axial_ecc'
                ,'Axial angle':'axial_angle'
                # ,'Planar component 1':'planar_component_1'
                # ,'Planar component 2':'planar_component_2'
                ,'Relative nuclear height':'rel_nuc_height'
                ,'Surface area':'sa'
                # ,'SA to vol':'ratio_sa_vol'
                # ,'Time to G1S':'time_g1s'
                ,'Basal area':'basal'
                ,'Apical area':'apical'
                
                # Growth rates
                ,'Specific GR b (sm)':'sgr'
                ,'Height to BM':'height_to_bm'
                ,'Mean curvature':'mean_curve'
                # ,'Gaussian curvature':'gaussian_curve'
                
                # Neighbor topolgy and
                # ,'Coronal angle':'cor_angle'
                ,'Coronal density':'cor_density'
                ,'Cell alignment':'cell_align'
                ,'Mean neighbor nuclear volume':'mean_neighb_nuc_vol'
                ,'Mean neighbor dist':'mean_neighb_dist'
                ,'Neighbor mean height frame-1':'neighb_height_12h'
                ,'Neighbor mean height frame-2':'neighb_height_24h'
                # ,'Num diff neighbors':'neighb_diff'
                # ,'Num planar neighbors':'neighb_plan'
                ,'Collagen fibrousness':'col_fib'
                ,'Collagen alignment':'col_align'}

df_g1s = df_.loc[:,list(features_list.keys())]
df_g1s = df_g1s.rename(columns=features_list)
df_g1s['G1S_logistic'] = (df_['Phase'] == 'SG2').astype(int)

df_g1s_test = df_test_.loc[:,list(features_list.keys())]
df_g1s_test = df_g1s_test.rename(columns=features_list)
df_g1s_test['G1S_logistic'] = (df_test_['Phase'] == 'SG2').astype(int)

# Standardize
for col in df_g1s.columns:
    df_g1s[col] = z_standardize(df_g1s[col])

for col in df_g1s_test.columns:
    df_g1s_test[col] = z_standardize(df_g1s_test[col])

# Count NANs
print(np.isnan(df_g1s).sum(axis=0))

#%% Robust LM for smoothed specific growth rate

from numpy.linalg import eig

############### OLS for specific growth rate ###############
model_rlm = smf.rlm(f'sgr ~ ' + str.join(' + ',
                                      df_g1s.columns.drop(['sgr'])),data=df_g1s).fit()
print(model_rlm.summary())

# model_rlm_ridge = smf.rlm(f'sgr ~ ' + str.join(' + ',
#                                       df_g1s.columns.drop(['sgr'])),data=df_g1s).fit_regularized('lasso')

############### GLM for specific growth rate ###############
# model_glm = smf.glm(f'sgr ~ ' + str.join(' + ',
#                                       df_g1s.columns[(df_g1s.columns != 'sgr') &
#                                                      (df_g1s.columns != 'gr')]),data=df_g1s).fit()
# print(model_glm.summary())
C = model_rlm.cov_params()
sb.heatmap(C,xticklabels=True,yticklabels=True)
L,D = eig(C)

print(f'Covariance eigenvalue ratio: {L.max()/L.min()}')

plt.figure()

plt.scatter(model_rlm.params[model_rlm.params > 0],-np.log10(model_rlm.pvalues[model_rlm.params > 0]),color='b')
plt.scatter(model_rlm.params[model_rlm.params < 0],-np.log10(model_rlm.pvalues[model_rlm.params < 0]),color='r')
sig_params = model_rlm.pvalues.index[model_rlm.pvalues < 0.05]
for p in sig_params:
    plt.text(model_rlm.params[p] + 0.01, -np.log10(model_rlm.pvalues[p]), p)

plt.hlines(-np.log10(0.05),xmin=-1,xmax=1,color='r')
plt.xlabel('Regression coefficient')
plt.ylabel('-Log(P)')

#%% Plot important parameters

from scipy.stats import stats

params = pd.DataFrame()

# Total corrcoef
X,Y = nonan_pairs(model_rlm.predict(df_g1s), df_g1s['sgr'])
R,P = stats.pearsonr(X,Y)
Rsqfull = R**2

params['var'] = model_rlm.params.index
params['coef'] = model_rlm.params.values
params['li'] = model_rlm.conf_int()[0].values
params['ui'] = model_rlm.conf_int()[1].values
params['pvals'] = model_rlm.pvalues.values

params['err'] = params['ui'] - params['coef'] 
params['effect size'] = np.sqrt(params['coef']**2 /(1-Rsqfull))

order = np.argsort( np.abs(params['coef']) )[::-1][0:10]
params = params.iloc[order]

plt.bar(range(len(params)),params['coef'],yerr=params['err'])
plt.ylabel('Regression coefficients')
plt.savefig('/Users/xies/Desktop/fig.eps')

#%% Cross-validation on the same dataset

from numpy import random

Niter = 100
N,P = df_g1s.shape
frac_withhold = 0.1

models = []
MSE = np.zeros(Niter)
Rsq = np.zeros(Niter)
Rsq_random = np.zeros(Niter)

coefficients = np.zeros((Niter,P-1))
pvalues = np.zeros((Niter,P-1))

for i in tqdm(range(Niter)):
    
    # Withold data
    num_withold = np.round(frac_withhold * N).astype(int)
    idx_subset = random.choice(N, size = num_withold, replace=False)
    Iwithheld = np.zeros(N).astype(bool)
    Iwithheld[idx_subset] = True
    Isubsetted = ~Iwithheld
    df_subsetted = df_g1s.loc[Isubsetted]
    df_withheld = df_g1s.loc[Iwithheld]
    
    this_model = smf.rlm(f'sgr ~ ' + str.join(' + ',df_subsetted.columns.drop('sgr')),
                         data=df_subsetted).fit()
    models.append(this_model)
    
    # predict on the withheld data
    ypred = this_model.predict(df_withheld)
    res = df_withheld['sgr'] - ypred
    MSE[i] = np.nansum( res ** 2 )

    R = np.corrcoef(*nonan_pairs(ypred, df_withheld['sgr']))[0,1]
    Rsq[i] = R**2
    
    
    # Generate a 'random' model
    df_rand = df_subsetted.copy()
    for col in df_rand.columns.drop('sgr'):
        df_rand[col] = random.randn(N-num_withold)
        
    random_model = smf.rlm(f'sgr ~ ' + str.join(' + ',df_rand.columns.drop('sgr')),
                           data=df_rand).fit()
    
    # predict on the withheld data
    ypred = random_model.predict(df_withheld)
    res = df_withheld['sgr'] - ypred
    MSE[i] = np.nansum( res ** 2 )
    
    R = np.corrcoef(*nonan_pairs(ypred, df_withheld['sgr']))[0,1]
    Rsq_random[i] = R**2
    
    coefficients[i,:] = this_model.params.drop('Intercept')
    pvalues[i,:] = this_model.pvalues.drop('Intercept')

coefficients = pd.DataFrame(coefficients,columns=df_g1s.columns.drop('sgr'))
pvalues = pd.DataFrame(pvalues,columns=df_g1s.columns.drop('sgr'))
    
# Plot R2
plt.hist(Rsq.flatten())
plt.hist(Rsq_random.flatten())
plt.legend(['Empirical','Random']),plt.xlabel('R2 on test set')

# Plot coefficient variance
plt.figure()
plt.errorbar(coefficients.mean(axis=0),y=-np.log10(pvalues).mean(axis=0),
             xerr = coefficients.std(axis=0),yerr= -np.log10(pvalues).std(axis=0),
             fmt='b*')
sig_params = coefficients.columns[pvalues.mean(axis=0) < 0.05]
for p in sig_params:
    plt.text(coefficients[p].mean() + 0.1, -np.log10(pvalues[p]).mean() + 0.01, p)
plt.hlines(-np.log10(0.05),xmin=-1,xmax=1,color='r')
plt.xlabel('Reg coefficient');plt.ylabel('-Log10(P)')

#%% Validate across regions

ypred = model_rlm.predict(df_g1s_test)
res = df_g1s_test['sgr'] - ypred
MSE_test = np.nansum( res**2 )

R_test = np.corrcoef(*nonan_pairs(ypred, df_g1s_test['sgr']))[0,1]

sb.regplot(df_g1s_test['sgr'],ypred)
plt.xlabel('Measured growth rates')
plt.ylabel('Predicted growth rates')


