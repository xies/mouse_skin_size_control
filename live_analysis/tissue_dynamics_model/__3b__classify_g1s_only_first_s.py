#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:14:21 2024

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb
# import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit
from basicUtils import *
from os import path
from tqdm import tqdm

from numpy import random
from sklearn.metrics import roc_curve, auc, confusion_matrix, average_precision_score
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

def rebalance_g1(df,Ng1):
    #% Rebalance class
    g1_sampled = df_g1s[df_g1s['G1S_logistic'] == 0].sample(Ng1,replace=False)
    # df_g1s[df_g1s['Phase' == 'G1']].sample
    sg2 = df_g1s[df_g1s['G1S_logistic'] == 1]

    df_g1s_balanced = pd.concat((g1_sampled,sg2),ignore_index=True)
    return df_g1s_balanced

def run_cross_validation(X,y,split_ratio,model,random_state=42):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=split_ratio,random_state=random_state)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    C = confusion_matrix(y_test,y_pred,normalize='all')
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    AUC = auc(fpr,tpr)
    AP = average_precision_score(y_test,y_pred)
    return C, AUC, AP

def z_standardize(x):
    return (x - np.nanmean(x))/np.std(x)

df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_g1s.csv',index_col=0)

collated_first_s = []
# Groupby CellID and then drop non-first S phase
for (cellID,frame),cell in df_g1s.groupby(['cellID','region']):
    if cell.G1S_logistic.sum() > 1:
        first_sphase_frame = np.where(cell.G1S_logistic)[0][0]
        collated_first_s.append(cell.iloc[0:first_sphase_frame+1])
    else:
        collated_first_s.append(cell)

df_g1s = pd.concat(collated_first_s)
df_g1s = df_g1s.drop(columns=['time_g1s','cellID','diff','region'])

Ng1 = 199

#%% Logistic for G1/S transition

Niter = 1000
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
             xerr = coefficients.std(axis=0),
             yerr = np.log10(pvalues).std(axis=0),
             fmt='bo')

# Label sig variables
sig_params = pvalues.columns[-np.log10(pvalues).mean(axis=0) > -np.log10(0.01)]
for p in sig_params:
    plt.text(coefficients[p].mean() + 0.01, -np.log10(pvalues[p]).mean() + 0.01, p)
plt.hlines(-np.log10(0.01),xmin=-1.5,xmax=2.0,color='r')
plt.xlabel('Regression coefficient')
plt.ylabel('-Log(P)')

params = pd.DataFrame()
params['var'] = coefficients.columns.values
params['coef'] = coefficients.mean(axis=0).values
params['li'] = li.mean(axis=0).values
params['ui'] = ui.mean(axis=0).values

params['err'] = params['ui'] - params['coef']
params['effect size'] = params['coef']

order = np.argsort( np.abs(params['coef']) )[::-1][0:5]
params = params.iloc[order]

plt.figure()
plt.bar(range(len(params)),params['coef'],yerr=params['err'])
plt.xticks(range(5),params['var'],rotation=30)

#%% Cross-validation for MLR

from sklearn.linear_model import LogisticRegression

Niter = 1000
frac_withheld = 0.1
N = len(df_g1s_balanced)

AUC = pd.DataFrame(np.zeros((Niter,2)),columns=['MLR','Random data'])
AP = pd.DataFrame(np.zeros((Niter,2)),columns=['MLR','Random data'])
C_mlr = np.zeros((Niter,2,2))
C_random = np.zeros((Niter,2,2))

for i in tqdm(range(Niter)):
    
    df_g1s_balanced = rebalance_g1(df_g1s,Ng1)
    y_balanced = df_g1s_balanced['G1S_logistic']
    df_g1s_balanced = df_g1s_balanced.drop(columns='G1S_logistic')

    #Logistic regression
    mlr = LogisticRegression(random_state = i)
    C_mlr[i,:,:],_AUC,_AP = run_cross_validation(df_g1s_balanced,y_balanced,frac_withheld,mlr)
    AUC.at[i,'MLR'] = _AUC
    AP.at[i,'MLR'] = _AP
    
    #Random data model
    df_random = pd.DataFrame(random.randn(*df_g1s_balanced.shape))
    random_model = LogisticRegression(random_state = i)
    C_random[i,:,:],_AUC,_AP = run_cross_validation(df_random,y_balanced,frac_withheld,random_model)
    AUC.at[i,'Random data'] = _AUC
    AP.at[i,'Random data'] = _AP
    
hist_weights = np.ones(Niter)/Niter
AUC.plot.hist(weights=hist_weights); plt.title(f'MLR classification cross-validation, {frac_withheld*100}% withheld');plt.xlabel('AUC')
plt.vlines(AUC['MLR'].mean(),0,0.8,'r')
AP.plot.hist(weights=hist_weights); plt.title(f'MLR classification cross-validation, {frac_withheld*100}% withheld');plt.xlabel('Average precision')
plt.vlines(AP['MLR'].mean(),0,0.8,'r')

plt.figure();sb.heatmap(np.mean(C_mlr,axis=0),xticklabels=['G1','SG2M'],yticklabels=['G1','SG2M'],annot=True)
plt.title(f'MLR Confusion matrix, {frac_withheld*100}% withheld, average over {Niter} iterations')

#%% Logistic: permutation importances

from sklearn.linear_model import LogisticRegression
frac_withheld = 0.1

g1_sampled = df_g1s[df_g1s['G1S_logistic'] == 0].sample(Ng1,replace=False)
sg2 = df_g1s[df_g1s['G1S_logistic'] == 1]
df_g1s_balanced = pd.concat((g1_sampled,sg2),ignore_index=True)
df_g1s_balanced['Random feature'] = random.randn(len(df_g1s_balanced))

X = df_g1s_balanced.drop(columns='G1S_logistic'); y = df_g1s_balanced['G1S_logistic']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=frac_withheld,random_state=42)
logist_model = LogisticRegression(random_state=42).fit(X_train,y_train)

result = permutation_importance(logist_model,X_test,y_test,n_repeats=1000,random_state=42,n_jobs=2)
logit_importances = pd.Series(result.importances_mean, index=X_train.columns).sort_values()

plt.figure()
logit_importances.plot.bar(yerr=result.importances_std)
plt.ylabel("Mean accuracy decrease")
plt.tight_layout()
plt.show()

#%% Random forest classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

Niter = 1000
frac_withheld = 0.2

AUC = pd.DataFrame(np.zeros((Niter,2)),columns=['RF','Random data'])
AP = pd.DataFrame(np.zeros((Niter,2)),columns=['RF','Random data'])
C_rf = np.zeros((Niter,2,2))
C_random = np.zeros((Niter,2,2))

for i in tqdm(range(Niter)):
    
    df_g1s_balanced = rebalance_g1(df_g1s,Ng1)
    y_balanced = df_g1s_balanced['G1S_logistic']
    
    # Random forest
    forest = RandomForestClassifier(n_estimators=100, random_state=i)
    C_rf[i,:,:],_AUC,_AP = run_cross_validation(df_g1s_balanced,y_balanced,frac_withheld,forest)
    AUC.at[i,'RF'] = _AUC
    AP.at[i,'RF'] = _AP
    
    # Random data model
    df_random = pd.DataFrame(random.randn(*df_g1s_balanced.shape))
    random_model = RandomForestClassifier(random_state = i)
    C_random[i,:,:],_AUC,_AP = run_cross_validation(df_random,y_balanced,frac_withheld,random_model)
    AUC.at[i,'Random data'] = _AUC
    AP.at[i,'Random data'] = _AP
    
    
hist_weights = np.ones(Niter)/Niter
AUC.plot.hist(bins=25,weights=hist_weights); plt.title(f'RF classification cross-validation, {frac_withheld*100}% withheld');plt.xlabel('AUC')
plt.vlines(AUC['RF'].mean(),0,0.8,'r')
AP.plot.hist(bins=25,weights=hist_weights); plt.title(f'RF classification cross-validation, {frac_withheld*100}% withheld');plt.xlabel('Average precision')
plt.vlines(AP['RF'].mean(),0,0.8,'r')

plt.figure();sb.heatmap(np.mean(C_rf,axis=0),xticklabels=['G1','SG2M'],yticklabels=['G1','SG2M'],annot=True)
plt.title(f'RF Confusion matrix, {frac_withheld*100}% withheld, average over {Niter} iterations')

#%% RF: Permutation importance

from sklearn.inspection import permutation_importance, partial_dependence

g1_sampled = df_g1s[df_g1s['G1S_logistic'] == 0].sample(Ng1,replace=False)
# df_g1s[df_g1s['Phase' == 'G1']].sample
sg2 = df_g1s[df_g1s['G1S_logistic'] == 1]    
df_g1s_balanced = pd.concat((g1_sampled,sg2),ignore_index=True)
df_g1s_balanced['Random feature'] = random.randn(len(df_g1s_balanced))

forest = RandomForestClassifier(n_estimators=100, random_state=i)

X = df_g1s_balanced.drop(columns='G1S_logistic'); y = df_g1s_balanced['G1S_logistic']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=frac_withhold,random_state=42)
forest.fit(X_train,y_train)
result = permutation_importance(forest,X_test,y_test,n_repeats=100,random_state=42,n_jobs=2)
forest_importances = pd.Series(result.importances_mean, index=X_train.columns)

top_forest_imp = forest_importances.iloc[forest_importances.argsort()][-10:][::-1]
top_forest_imp_std = result.importances_std[forest_importances.argsort()][-10:][::-1]
top_forest_imp.plot.bar(yerr=top_forest_imp_std)
plt.ylabel("Mean accuracy decrease")
plt.tight_layout()
plt.show()


#%%

# plt.figure()
# sb.barplot(data=imp.melt(value_vars=imp.columns),x='variable',y='value',order=imp.columns[imp.mean(axis=0).values.argsort()[::-1]]);
# plt.xticks(rotation=45);plt.ylabel('Importance')

import os
from sklearn import tree
# tree.plot_tree(rf_random.best_estimator_.estimators_[k])

plt.figure()
# Export as dot file
# tree.plot_tree(forest[0])
tree.plot_tree(forest.estimators_[0],
               feature_names = df_g1s.columns,
                     filled=True,
                     max_depth=1)


# param_dist = {'n_estimators': randint(50,500),
#               'max_depth': randint(1,20)}

# # Create a random forest classifier
# rf = RandomForestClassifier()

# # Use random search to find the best hyperparameters
# rand_search = RandomizedSearchCV(rf, 
#                                  param_distributions = param_dist, 
#                                  n_iter=5, 
#                                  cv=5)

# # Fit the random search object to the data
# rand_search.fit(X_train, y_train)


#%% Compare MLR v RF

plt.figure()
sb.heatmap(C_mlr.mean(axis=0),annot=True,xticklabels=['G1','SG2'],yticklabels=['Pred G1','Pred SG2'])
plt.title('Logistic')
plt.figure()
sb.heatmap(C_rf.mean(axis=0),annot=True,xticklabels=['G1','SG2'],yticklabels=['Pred G1','Pred SG2'])
plt.title('Rand Forest')
plt.savefig('/Users/xies/Desktop/rfm.eps')


