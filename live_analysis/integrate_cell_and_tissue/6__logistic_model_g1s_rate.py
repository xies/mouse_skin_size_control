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

df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_g1s.csv',index_col=0)

#%% Logistic for G1/S transition

Ng1 = 150
Niter = 50

coefficients = np.ones((Niter,df_g1s.shape[1]-1)) * np.nan
li = np.ones((Niter,df_g1s.shape[1]-1)) * np.nan
ui = np.ones((Niter,df_g1s.shape[1]-1)) * np.nan
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
    li[i,:] = model_g1s.conf_int()[0].drop('Intercept')
    ui[i,:] = model_g1s.conf_int()[1].drop('Intercept')
    pvalues[i,:] = pvals.values

coefficients = pd.DataFrame(coefficients,columns = df_g1s.columns.drop('G1S_logistic')).dropna()
pvalues = pd.DataFrame(pvalues,columns = df_g1s.columns.drop('G1S_logistic')).dropna()
li = pd.DataFrame(li,columns = df_g1s.columns.drop('G1S_logistic')).dropna()
ui = pd.DataFrame(ui,columns = df_g1s.columns.drop('G1S_logistic')).dropna()

# Effect size v. pvalue
plt.errorbar(x=coefficients.mean(axis=0), y=-np.log10(pvalues).mean(axis=0),
             xerr = coefficients.std(axis=0)/np.sqrt(Niter),
             yerr = -np.log10(pvalues).std(axis=0)/np.sqrt(Niter),
             fmt='bo')
# Label sig variables
sig_params = pvalues.columns[-np.log10(pvalues).mean(axis=0) > -np.log10(0.05)]
for p in sig_params:
    plt.text(coefficients[p].mean() + 0.1, -np.log10(pvalues[p]).mean() + 0.01, p)
plt.hlines(-np.log10(0.05),xmin=-1.5,xmax=2.0,color='r')
plt.xlabel('Regression coefficient')
plt.ylabel('-Log(P)')

params = pd.DataFrame()
params['var'] = coefficients.columns.values
params['coef'] = coefficients.mean(axis=0).values
params['li'] = li.mean(axis=0).values
params['ui'] = ui.mean(axis=0).values

params['err'] = params['ui'] - params['coef']
params['effect size'] = np.sqrt(params['coef']**2 /(1-Rsqfull))

order = np.argsort( np.abs(params['coef']) )[::-1][0:10]
params = params.iloc[order]

plt.figure()
plt.bar(range(len(params)),params['coef'],yerr=params['err'])
plt.xticks(range(10),params['var'],rotation=30)

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
C_mlr = np.zeros((Niter,2,2))
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
    
    C_mlr[i,:,:] = confusion_matrix(labels,ypred>0.5,normalize='all')
    
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

# #%% Plot confusion matrix as bar

# g1_rates = C[:,0,0]
# sg2_rates = C[:,1,1]
# g1_rates_random = C_random[:,0,0]
# sg2_rates_random = C_random[:,1,1]

# rates = pd.DataFrame(columns = ['accuracy','phase','random'])
# rates['accuracy'] = np.hstack((g1_rates,g1_rates_random,sg2_rates,sg2_rates_random))
# rates.loc[0:200,'phase'] = 'G1'
# rates.loc[200:400,'phase'] = 'SG2'
# rates.loc[0:100,'random'] = False
# rates.loc[100:200,'random'] = True
# rates.loc[200:300,'random'] = False
# rates.loc[300:400,'random'] = True

# sb.catplot(data = rates,x='phase',y='accuracy',kind='violin',hue='random',split=True)
# # plt.ylabel('True positive rate')

# #%% Generate PR curve based on test dataset

# ypred = model_g1s.predict(df_g1s_test)

# IdropNA = ~np.isnan(ypred.values)
# ypred = ypred.values[IdropNA]
# labels = df_g1s_test['G1S_logistic'].values[IdropNA]

# precision,recall,th = metrics.precision_recall_curve(labels,ypred)
# metrics.PrecisionRecallDisplay(precision=precision, recall=recall).plot()
# AP = metrics.average_precision_score(labels,ypred)

# fpr, tpr, _ = metrics.roc_curve(labels, ypred)
# metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

# AUC_test = metrics.auc(fpr,tpr)


#%% Random forest classifier

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

Niter = 100
sum_res = np.zeros(Niter)
Rsq = np.zeros(Niter)
importance = np.zeros((Niter,df_g1s.shape[1]))
AUC = np.zeros(Niter); AP = np.zeros(Niter)
C_rf = np.zeros((Niter,2,2))

for i in tqdm(range(Niter)):
    
    g1_sampled = df_g1s[df_g1s['G1S_logistic'] == 0].sample(Ng1,replace=False)
    # df_g1s[df_g1s['Phase' == 'G1']].sample
    sg2 = df_g1s[df_g1s['G1S_logistic'] == 1]    
    df_g1s_balanced = pd.concat((g1_sampled,sg2),ignore_index=True)
    df_g1s_balanced['Random feature'] = random.randn(len(df_g1s_balanced))
    
    forest = RandomForestClassifier(n_estimators=100, random_state=i)
    
    X = df_g1s_balanced.drop(columns='G1S_logistic'); y = df_g1s_balanced['G1S_logistic']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)
    forest.fit(X_train,y_train)
    
    y_pred = forest.predict(X_test)
    importance[i,:] = forest.feature_importances_
    
    C_rf[i,:,:] = confusion_matrix(y_test,y_pred,normalize='all')
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    AUC[i] = metrics.auc(fpr,tpr)
    AP[i] = metrics.average_precision_score(y_test,y_pred)
    
plt.hist(AUC); plt.hist(AP)

imp = pd.DataFrame(importance)
imp.columns = df_g1s_balanced.columns.drop('G1S_logistic')

#%%

plt.figure()
sb.barplot(data=imp.melt(value_vars=imp.columns),x='variable',y='value',order=imp.columns[imp.mean(axis=0).values.argsort()[::-1]]);
plt.xticks(rotation=45);plt.ylabel('Importance')

#%% Compare MLR v RF

# plt.figure()
# sb.heatmap(C_mlr.mean(axis=0),annot=True,xticklabels=['G1','SG2'],yticklabels=['Pred G1','Pred SG2'])
# plt.title('Logistic')
plt.figure()
sb.heatmap(C_rf.mean(axis=0),annot=True,xticklabels=['G1','SG2'],yticklabels=['Pred G1','Pred SG2'])
plt.title('Rand Forest')
plt.savefig('/Users/xies/Desktop/rfm.eps')


