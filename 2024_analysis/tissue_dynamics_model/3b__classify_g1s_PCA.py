#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:04:34 2022

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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def keep_only_first_sg2(df):
    collated_first_s = []
    # Groupby CellID and then drop non-first S phase
    for (cellID,frame),cell in df.groupby(['cellID','region']):
        if cell.G1S_logistic.sum() > 1:
            first_sphase_frame = np.where(cell.G1S_logistic)[0][0]
            collated_first_s.append(cell.iloc[0:first_sphase_frame+1])
        else:
            collated_first_s.append(cell)
    return pd.concat(collated_first_s,ignore_index=True)

def run_pca(df,ncomp):
    
    pca = PCA(n_components = Ncomp)
    X_ = pca.fit_transform(df)
    df_pca = pd.DataFrame(X_,columns = [f'PC{str(i)}' for i in range(Ncomp)])
    components = pd.DataFrame(pca.components_,columns=df.columns)
    
    return df_pca, components, pca

def plot_principle_component(loadings,which_comp):
    loadings = loadings.iloc[which_comp]
    loadings = loadings[np.abs(loadings).argsort()[::-1]]
    plt.figure()
    np.abs(loadings).plot.bar()
    plt.ylabel('Magnitude of loading')
    plt.xlabel('Original dimensions');plt.title(f'PC {which_comp}')
    plt.tight_layout()
    
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


df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_g1s_final.csv',index_col=0)
df_g1s = keep_only_first_sg2(df_g1s)

df_g1s = df_g1s.drop(columns=['time_g1s','cellID','diff','fucci_int_12h'])
y = df_g1s['G1S_logistic']

#%% PCA on unbalanced data

Ncomp = 20

df_pca,loadings, pca = run_pca(df_g1s.drop(columns='G1S_logistic'),Ncomp)

plt.figure()
plt.bar(np.arange(1,Ncomp+1),pca.explained_variance_ratio_); plt.ylabel('% variance explained');plt.xlabel('Components')
plt.figure()
plt.bar(np.arange(1,Ncomp+1),np.cumsum(pca.explained_variance_ratio_)); plt.ylabel('Cumulative % variance explained');plt.xlabel('Components')



plot_principle_component(loadings,0)
plot_principle_component(loadings,1)
plot_principle_component(loadings,2)
plot_principle_component(loadings,3)

plot_principle_component(loadings,6)
plot_principle_component(loadings,17)

#%% PCA regression

Ng1 = 150
Ncomp = 20
Niter = 100
frac_withheld = 0.1

AP = pd.DataFrame()
AUC = pd.DataFrame()

C_mlr = np.zeros((Niter,2,2))
C_random = np.zeros((Niter,2,2))

from sklearn.linear_model import LogisticRegression

for i in range(Niter):
    
    df_g1s_balanced = rebalance_g1(df_g1s,Ng1)
    y_balanced = df_g1s_balanced['G1S_logistic']

    df_pca,_,_ = run_pca(df_g1s_balanced.drop(columns='G1S_logistic'),Ncomp)
    
    mlr = LogisticRegression(random_state = i)
    C_mlr[i,:,:],auc_,ap = run_cross_validation(df_pca,y_balanced,frac_withheld,mlr)
    AUC.at[i,'Full'] = auc_
    AP.at[i,'Full'] = ap
    
    # Only irst PC
    mlr_only0 = LogisticRegression(random_state = i)
    C_mlr[i,:,:],auc_,ap = run_cross_validation(df_pca[['PC0']],y_balanced,frac_withheld,mlr_only0)
    AUC.at[i,'Only PC0'] = auc_
    AP.at[i,'Only PC0'] = ap
    
    df_random = pd.DataFrame(random.randn(len(df_pca),Ncomp))
    random_model = LogisticRegression(random_state = i)
    C_random[i,:,:],auc_,ap = run_cross_validation(df_random,y_balanced,frac_withheld,random_model)
    AUC.at[i,'Random'] = auc_
    AP.at[i,'Random'] = ap

sb.histplot(AUC.melt(),x='value',hue='variable',element='poly',bins=15,stat='probability')
plt.xlabel('AUC')
AUC.to_excel('/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/Tissue model/PCA/Classify G1S single SG2 logistic/AUCs.xlsx')
 
plt.figure();sb.heatmap(np.mean(C_mlr,axis=0),xticklabels=['G1','SG2M'],yticklabels=['G1','SG2M'],annot=True)
plt.title(f'Confusion matrix, {frac_withheld*100}% withheld, average over {Niter} iterations')

#%% MLR: Permutation importance

from sklearn.inspection import permutation_importance

df_g1s_balanced = rebalance_g1(df_g1s,Ng1)
y_balanced = df_g1s_balanced['G1S_logistic']
df_pca,_,_ = run_pca(df_g1s_balanced.drop(columns='G1S_logistic'),Ncomp)
X_train,X_test,y_train,y_test = train_test_split(df_pca,y_balanced,test_size=frac_withheld,random_state=42)

mlr = LogisticRegression(random_state=42)
mlr.fit(X_train,y_train)
result = permutation_importance(mlr,X_test,y_test,n_repeats=100,random_state=42,n_jobs=2)
forest_importances = pd.Series(result.importances_mean, index=X_train.columns)

imps = pd.DataFrame(forest_importances)
imps['std'] = result.importances_std
imps.to_excel('/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/Tissue model/PCA/Classify G1S single SG2 logistic/importances.xlsx')

top_forest_imp = forest_importances.iloc[forest_importances.argsort()][-10:][::-1]
top_forest_imp_std = result.importances_std[forest_importances.argsort()][-10:][::-1]
top_forest_imp.plot.bar(yerr=top_forest_imp_std)
plt.ylabel("Mean accuracy decrease")
plt.xlabel('Permuted feature')
plt.tight_layout()
plt.show()

#%% Random forest classifier

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn import metrics

Ncomp = 20
Niter = 1000

frac_withheld = 0.1
sum_res = np.zeros(Niter)
Rsq = np.zeros(Niter)
AUC_rf = np.zeros(Niter); AP_rf = np.zeros(Niter)
AUC_random = np.zeros(Niter); AP_random = np.zeros(Niter)
C_rf = np.zeros((Niter,2,2))
C_random = np.zeros((Niter,2,2))

for i in tqdm(range(Niter)):
    
    df_g1s_balanced = rebalance_g1(df_g1s,Ng1)
    y_balanced = df_g1s_balanced['G1S_logistic']
    df_pca,_,_ = run_pca(df_g1s_balanced.drop(columns='G1S_logistic'),Ncomp)

    forest = RandomForestClassifier(n_estimators=100, random_state=i)
    
    C_rf[i,:,:],AUC_rf[i], AP_rf[i] = run_cross_validation(df_pca,y_balanced,frac_withheld,forest)

    # Generate a 'random' model
    df_random = pd.DataFrame(random.randn(len(df_pca),Ncomp))
    random_model = RandomForestClassifier(n_estimators=100, random_state=i)
    C_random[i,:,:],AUC_random[i], AP_random[i] = run_cross_validation(df_random,y_balanced,frac_withheld,random_model)
    
hist_weights = np.ones(Niter)/Niter
plt.figure();plt.hist(AUC_rf,weights=hist_weights)
plt.xlabel('AUC');plt.title(f'RF classification cross-validation, {frac_withheld*100}%withheld')
plt.figure();plt.hist(AP_rf,weights=hist_weights)
plt.xlabel('Average precision');plt.title(f'RF classification cross-validation, {frac_withheld*100}% withheld')
    
plt.figure();sb.heatmap(np.mean(C_rf,axis=0),xticklabels=['G1','SG2M'],yticklabels=['G1','SG2M'],annot=True)
plt.title(f'Confusion matrix, {frac_withheld*100}% withheld, average over {Niter} iterations')

#%% RF: Permutation importance

df_g1s_balanced = rebalance_g1(df_g1s,Ng1)
y_balanced = df_g1s_balanced['G1S_logistic']
df_pca,_,_ = run_pca(df_g1s_balanced.drop(columns='G1S_logistic'),Ncomp)
X_train,X_test,y_train,y_test = train_test_split(df_pca,y_balanced,test_size=frac_withheld,random_state=42)

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train,y_train)
result = permutation_importance(forest,X_test,y_test,n_repeats=100,random_state=42,n_jobs=2)
forest_importances = pd.Series(result.importances_mean, index=X_train.columns)

top_forest_imp = forest_importances.iloc[forest_importances.argsort()][-10:][::-1]
top_forest_imp_std = result.importances_std[forest_importances.argsort()][-10:][::-1]
top_forest_imp.plot.bar(yerr=top_forest_imp_std)
plt.ylabel("Mean accuracy decrease")
plt.xlabel('Permuted feature')
plt.tight_layout()
plt.show()

