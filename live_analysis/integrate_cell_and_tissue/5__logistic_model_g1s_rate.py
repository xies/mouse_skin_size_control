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
from tqdm import tqdm

from numpy import random
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import scale 
from sklearn.cross_decomposition import PLSCanonical
from sklearn.covariance import EmpiricalCovariance

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

#%%

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
df1 = pd.read_csv(path.join(dirname,'MLR model/ts_features.csv'),index_col=0)
df1_ = df1[df1['Phase'] != '?']

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'
df2 = pd.read_csv(path.join(dirname,'MLR model/ts_features.csv'),index_col=0)
df2_ = df2[df2['Phase'] != '?']

df_ = pd.concat((df1_,df2_),ignore_index=True)
# df_ = df_1
N,P = df_.shape

#%% Sanitize field names for smf

features_list = { # Cell geometry
                'Age':'age'
                # ,'Z_x':'z','Y_x':'y','X_x':'x'
                ,'Volume (sm)':'vol_sm'
                # ,'Axial component':'axial_moment'
                ,'Nuclear volume':'nuc_vol'
                ,'Nuclear surface area':'nuc_sa'
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
                
                # Neighbor topolgy and geometry
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

# Standardize
for col in df_g1s.columns:
    df_g1s[col] = z_standardize(df_g1s[col])
    
for col in df_g1s_test.columns:
    df_g1s_test[col] = z_standardize(df_g1s_test[col])

df_g1s['G1S_logistic'] = (df_['Phase'] == 'SG2').astype(float)

print(df_g1s.isnull().sum(axis=0))
df_g1s = df_g1s.dropna()

C = EmpiricalCovariance().fit(df_g1s.drop(columns='G1S_logistic').dropna())
sb.heatmap(C.covariance_)
L,D = eig(C.covariance_); print(L.max()/L.min())

#%% Logistic for G1/S transition

Ng1 = 150
Niter = 1000

coefficients = np.ones((Niter,df_g1s.shape[1]-1)) * np.nan
pvalues = np.ones((Niter,df_g1s.shape[1]-1)) * np.nan

for i in range(Niter):
    
    #% Rebalance class
    g1_sampled = df_g1s[df_g1s['G1S_logistic'] == 0].sample(Ng1,replace=False)
    # df_g1s[df_g1s['Phase' == 'G1']].sample
    sg2 = df_g1s[df_g1s['G1S_logistic'] == 1]
    
    df_g1s_balanced = pd.concat((g1_sampled,sg2),ignore_index=True)
    
    ############### G1S logistic as function of age ###############
    try:
        model_g1s = smf.logit('G1S_logistic ~ ' + str.join(' + ',df_g1s.columns[df_g1s.columns != 'G1S_logistic']),
                      data=df_g1s_balanced).fit()
    except:
        continue
    
    # plt.figure()
    params = model_g1s.params.drop('Intercept')
    pvals = model_g1s.pvalues.drop('Intercept')

    coefficients[i,:] = params.values
    pvalues[i,:] = pvals.values

coefficients = pd.DataFrame(coefficients,columns = df_g1s.columns.drop('G1S_logistic')).dropna()
pvalues = pd.DataFrame(pvalues,columns = df_g1s.columns.drop('G1S_logistic')).dropna()


plt.errorbar(x=coefficients.mean(axis=0), y=-np.log10(pvalues).mean(axis=0),
             xerr = coefficients.std(axis=0)/np.sqrt(Niter),
             yerr = -np.log10(pvalues).std(axis=0)/np.sqrt(Niter),
             fmt='b*')

sig_params = params.index[-np.log10(pvalues).mean(axis=0) > -np.log10(0.05)]
for p in sig_params:
    plt.text(coefficients[p].mean() + 0.1, -np.log10(pvalues[p]).mean() + 0.01, p)

plt.hlines(-np.log10(0.05),xmin=-1.5,xmax=2.0,color='r')
plt.xlabel('Regression coefficient')
plt.ylabel('-Log(P)')

#%%

from scipy.stats import stats

params = pd.DataFrame()

# Total corrcoef
X,Y = nonan_pairs(model_g1s.predict(df_g1s), df_g1s['sgr'])
R,P = stats.pearsonr(X,Y)
Rsqfull = R**2

params['var'] = model_g1s.params.index
params['coef'] = model_g1s.params.values
params['li'] = model_g1s.conf_int()[0].values
params['ui'] = model_g1s.conf_int()[1].values
params['pval'] = model_g1s.pvalues.values

params['err'] = params['ui'] - params['coef'] 

params['effect size'] = np.sqrt(params['coef']**2 /(1-Rsqfull))

order = np.argsort( np.abs(params['coef']) )[::-1][1:11]
params = params.iloc[order]

plt.bar(range(len(params)),params['coef'],yerr=params['err'])
plt.ylabel('Regression coefficients')
plt.savefig('/Users/xies/Desktop/fig.eps')

#%% Cross-validation

from numpy import random
from sklearn import metrics

Niter = 100

frac_withhold = 0.1
N = len(df_g1s_balanced)

models = []
random_models = []
AUC = np.zeros(Niter)
AP = np.zeros(Niter)
C = np.zeros((Niter,2,2))
AUC_random= np.zeros(Niter)
AP_random = np.zeros(Niter)
C_random = np.zeros((Niter,2,2))

for i in tqdm(range(Niter)):
    
    #% Rebalance class
    g1_sampled = df_g1s[df_g1s['G1S_logistic'] == 0].sample(Ng1,replace=False)
    # df_g1s[df_g1s['Phase' == 'G1']].sample
    sg2 = df_g1s[df_g1s['G1S_logistic'] == 1]
    
    df_g1s_balanced = pd.concat((g1_sampled,sg2),ignore_index=True)
    
    num_withold = np.round(frac_withhold * N).astype(int)
    
    idx_subset = random.choice(N, size = num_withold, replace=False)
    Iwithheld = np.zeros(N).astype(bool)
    Iwithheld[idx_subset] = True
    Isubsetted = ~Iwithheld
    
    df_subsetted = df_g1s_balanced.loc[Isubsetted]
    df_withheld = df_g1s_balanced.loc[Iwithheld]
    
    this_model = smf.logit('G1S_logistic ~ ' + str.join(' + ',df_subsetted.columns[df_subsetted.columns != 'G1S_logistic']),
                  data=df_subsetted).fit()
    models.append(this_model)
    
    # Generate a 'random' model
    df_rand = df_subsetted.copy()
    for col in df_rand.columns.drop('G1S_logistic'):
        df_rand[col] = random.randn(N-num_withold)
        
    random_model = smf.logit('G1S_logistic ~ ' + str.join(' + ',df_rand.columns[df_rand.columns != 'G1S_logistic']),
                  data=df_rand).fit()
    random_models.append(random_model)
    
    # predict on the withheld data
    ypred = this_model.predict(df_withheld).values
    IdropNA = ~np.isnan(ypred)
    ypred = ypred[IdropNA]
    labels = df_withheld['G1S_logistic'].values[IdropNA]
    
    fpr, tpr, _ = metrics.roc_curve(labels, ypred)
    AUC[i] = metrics.auc(fpr,tpr)
    
    precision,recall,th = metrics.precision_recall_curve(labels,ypred)
    AP[i] = metrics.average_precision_score(labels,ypred)
    
    C[i,:,:] = confusion_matrix(labels,ypred>0.5,normalize='all')
    
    # predict on the withheld data
    ypred = random_model.predict(df_withheld).values
    IdropNA = ~np.isnan(ypred)
    ypred = ypred[IdropNA]
    labels = df_withheld['G1S_logistic'].values[IdropNA]
    
    fpr, tpr, _ = metrics.roc_curve(labels, ypred)
    AUC_random[i] = metrics.auc(fpr,tpr)
    
    precision,recall,th = metrics.precision_recall_curve(labels,ypred)
    AP_random[i] = metrics.average_precision_score(labels,ypred)
    
    C_random[i,:,:] = confusion_matrix(labels,ypred>0.5,normalize='all')
    
    
plt.hist(AUC)
plt.hist(AP)

#%% Plot confusion matrix as bar

g1_rates = C[:,0,0]
sg2_rates = C[:,1,1]
g1_rates_random = C_random[:,0,0]
sg2_rates_random = C_random[:,1,1]

rates = pd.DataFrame(columns = ['accuracy','phase','random'])
rates['accuracy'] = np.hstack((g1_rates,g1_rates_random,sg2_rates,sg2_rates_random))
rates.loc[0:200,'phase'] = 'G1'
rates.loc[200:400,'phase'] = 'SG2'
rates.loc[0:100,'random'] = False
rates.loc[100:200,'random'] = True
rates.loc[200:300,'random'] = False
rates.loc[300:400,'random'] = True

sb.catplot(data = rates,x='phase',y='accuracy',kind='violin',hue='random',split=True)
# plt.ylabel('True positive rate')

#%% Generate PR curve based on test dataset

ypred = model_g1s.predict(df_g1s_test)

IdropNA = ~np.isnan(ypred.values)
ypred = ypred.values[IdropNA]
labels = df_g1s_test['G1S_logistic'].values[IdropNA]

precision,recall,th = metrics.precision_recall_curve(labels,ypred)
metrics.PrecisionRecallDisplay(precision=precision, recall=recall).plot()
AP = metrics.average_precision_score(labels,ypred)

fpr, tpr, _ = metrics.roc_curve(labels, ypred)
metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

AUC_test = metrics.auc(fpr,tpr)


