#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 18:18:35 2022

@author: xies
"""


import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from basicUtils import *
from os import path
from tqdm import tqdm

from numpy import random
from sklearn.metrics import roc_curve, auc, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale 
from scipy.stats import stats
from sklearn.ensemble import RandomForestClassifier

def z_standardize(x):
    return (x - np.nanmean(x))/np.std(x)

def rebalance_g1(df,Ng1):
    #% Rebalance class
    g1_sampled = df_g1s[df_g1s['G1S_logistic'] == 0].sample(Ng1,replace=False)
    # df_g1s[df_g1s['Phase' == 'G1']].sample
    sg2 = df_g1s[df_g1s['G1S_logistic'] == 1]

    df_g1s_balanced = pd.concat((g1_sampled,sg2),ignore_index=True)
    return df_g1s_balanced

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

def run_cross_validation(X,y,split_ratio,model,random_state=42):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=split_ratio,random_state=random_state)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    C = confusion_matrix(y_test,y_pred,normalize='all')
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    AUC = auc(fpr,tpr)
    AP = average_precision_score(y_test,y_pred)
    return C, AUC, AP

df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_.csv',index_col=0)
df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_g1s.csv',index_col=0)

df_g1s = keep_only_first_sg2(df_g1s)

df_g1s = df_g1s.drop(columns=['time_g1s','fucci_int_12h','cellID','diff'])

Ng1 = 150

#%% Recursive feature drop

'''

MULTI-LOGISTIC REGRESSION

'''

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

X = df_g1s.drop(columns='G1S_logistic')
y = df_g1s['G1S_logistic']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)

feature2drop = []
selector = RFE(LogisticRegression(max_iter=1000), n_features_to_select=1)
selector = selector.fit(X_train,y_train)
features_ranked_by_RFE = X_train.columns.values[selector.ranking_.argsort()]
print(features_ranked_by_RFE[:10])

#%%

from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

df_g1s_balanced = rebalance_g1(df_g1s,Ng1)
y_balanced = df_g1s_balanced['G1S_logistic']

X = df_g1s_balanced.drop(columns='G1S_logistic'); y = df_g1s_balanced['G1S_logistic']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=frac_withheld,random_state=42)
logist_model = LogisticRegression(random_state=42,max_iter=100).fit(X_train,y_train)
result = permutation_importance(logist_model,X_test,y_test,n_repeats=100,random_state=42,n_jobs=2)
logit_importances = pd.Series(result.importances_mean, index=X_train.columns).sort_values(ascending=False)

plt.figure()
logit_importances.plot.bar(yerr=result.importances_std)
plt.ylabel("Mean accuracy decrease")
plt.tight_layout()
plt.show()

#%% MLR: AUC for volume feature drop v. every other feature

Niter = 100
frac_withheld = 0.1

features2drop = logit_importances.index[1:]
AUC = pd.DataFrame()
AP = pd.DataFrame()

for i in tqdm(range(Niter)):
    
    df_g1s_balanced = rebalance_g1(df_g1s,Ng1)
    y_balanced = df_g1s_balanced['G1S_logistic']
    df_g1s_balanced = df_g1s_balanced.drop(columns='G1S_logistic')
    X = df_g1s_balanced
    
    mlr = LogisticRegression(max_iter=1000, random_state=42)
    _, _AUC,_AP = run_cross_validation(df_g1s_balanced,y_balanced,frac_withheld,mlr)
    AUC.at[i,'Full'] = _AUC; AP.at[i,'Full'] = _AP
    
    mlr_no_vol = LogisticRegression(max_iter=1000, random_state=42)
    _, _AUC,_AP = run_cross_validation(df_g1s_balanced.drop(columns='vol_sm'),y_balanced,frac_withheld,mlr_no_vol)
    AUC.at[i,'No vol'] = _AUC; AP.at[i,'No vol'] = _AP
    
    df_random = pd.DataFrame(random.randn(*df_g1s_balanced.shape))
    random_model = LogisticRegression(max_iter=1000, random_state=42)
    _,_AUC,_AP = run_cross_validation(df_random,y_balanced,frac_withheld,random_model)
    AUC.at[i,'Random'] = _AUC; AP.at[i,'Random'] = _AP
    
    for j in range(len(features2drop)):
        model_no_other = LogisticRegression(max_iter=1000, random_state=42)
        _, _AUC,_AP = run_cross_validation(df_g1s_balanced.drop(columns=features2drop[j]),y_balanced,frac_withheld,model_no_other)
        AUC.at[i,features2drop[j]] = _AUC; AP.at[i,features2drop[j]] = _AP

I = AUC.mean().sort_values().index
sb.barplot(AUC[I])
plt.ylim([0.4,1]);plt.ylabel('Mean AUC');
plt.hlines(AUC.mean()['Full'],0,13)
plt.xticks(rotation=90);plt.tight_layout();

AUC_ = AUC.melt()
AUC_['Category'] = AUC_['variable']
AUC_.loc[(AUC_['variable'] != 'Full') & (AUC_['variable'] != 'Random') & (AUC_['variable'] != 'No vol'),'Category'] = 'No other'
AUC_.groupby('Category').mean()
# AUC_ = AUC_[AUC_['Category'] != 'Random']

plt.figure()
sb.histplot(AUC_,x='value',hue='Category',bins = 10,stat='probability',element='poly',common_norm=False,fill=False)
plt.vlines(AUC_.groupby('Category').mean(),0,0.25)

#%% Single features; MLR

Niter = 100

features2drop = logit_importances.index[1:]
df_g1s['Intercept'] = 1

AUC = pd.DataFrame()
AP = pd.DataFrame()

for i in tqdm(range(Niter)):
    
    df_g1s_balanced = rebalance_g1(df_g1s,Ng1)
    y_balanced = df_g1s_balanced['G1S_logistic']
    df_g1s_balanced = df_g1s_balanced.drop(columns='G1S_logistic')
    
    forest = LogisticRegression(max_iter=1000, random_state=42)
    _, _AUC,_AP = run_cross_validation(df_g1s_balanced,y_balanced,frac_withheld,forest)
    AUC.at[i,'Full'] = _AUC; AP.at[i,'Full'] = _AP
    
    forest_no_vol = LogisticRegression(max_iter=1000, random_state=42)
    _, _AUC,_AP = run_cross_validation(df_g1s_balanced[['vol_sm','Intercept']],y_balanced,frac_withheld,forest_no_vol)
    AUC.at[i,'Only vol'] = _AUC; AP.at[i,'Only vol'] = _AP
    
    df_random = pd.DataFrame(random.randn(*df_g1s_balanced.shape))
    random_model = LogisticRegression(max_iter=1000, random_state=42)
    _,_AUC,_AP = run_cross_validation(df_random,y_balanced,frac_withheld,random_model)
    AUC.at[i,'Random'] = _AUC; AP.at[i,'Random'] = _AP
    
    for j in range(len(features2drop)):
        forest = LogisticRegression(max_iter=1000, random_state=42)
        _, _AUC,_AP = run_cross_validation(df_g1s_balanced[['Intercept',features2drop[j]]],y_balanced,frac_withheld,forest)
        AUC.at[i,f'Only {features2drop[j]}'] = _AUC; AP.at[i,f'No {features2drop[j]}'] = _AP
        
I = AUC.mean().sort_values().index
sb.barplot(AUC[I])
plt.ylim([0.4,1]);plt.ylabel('Mean AUC');
plt.hlines(AUC.mean()['Full'],0,40)
plt.xticks(rotation=90);plt.tight_layout();

AUC_ = pd.melt(AUC)
AUC_['Category'] = AUC_['variable']
AUC_.loc[(AUC_['variable']!= 'Full')&(AUC_['variable']!= 'Only vol')&(AUC_['variable']!= 'Random'),'Category'] = 'Other'

plt.figure()
sb.histplot(AUC_,bins=20,x='value',hue='Category',stat='probability',element='poly',common_norm=False,fill=False)
plt.vlines(AUC_.groupby('Category').mean(),0,0.25)

#%% Recursive feature drop, Random Forest

'''

RANDOM FOREST

'''


from sklearn.feature_selection import RFE

X = df_g1s.drop(columns='G1S_logistic')
X['Intercept'] = 1
y = df_g1s['G1S_logistic']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)

feature2drop = []
selector = RFE(RandomForestClassifier(n_estimators=100), n_features_to_select=1)
selector = selector.fit(X_train,y_train)
features_ranked_by_RFE = X_train.columns.values[selector.ranking_.argsort()]
print(features_ranked_by_RFE[:10])

#%% RF: Permutation importance

df_g1s_balanced = rebalance_g1(df_g1s,Ng1)
y_balanced = df_g1s_balanced['G1S_logistic']

forest = RandomForestClassifier(n_estimators=100, random_state=i)
X = df_g1s_balanced.drop(columns='G1S_logistic'); y = df_g1s_balanced['G1S_logistic']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=frac_withheld,random_state=42)
forest.fit(X_train,y_train)
result = permutation_importance(forest,X_test,y_test,n_repeats=100,random_state=42,n_jobs=2)
forest_importances = pd.Series(result.importances_mean, index=X_train.columns)

top_forest_imp = forest_importances.iloc[forest_importances.argsort()][-10:][::-1]
top_forest_imp_std = result.importances_std[forest_importances.argsort()][-10:][::-1]
top_forest_imp.plot.bar(yerr=top_forest_imp_std)
plt.ylabel("Mean accuracy decrease")
plt.tight_layout()
plt.show()

#%% Random forest: AUC for volume feature drop v. every other feature

Ng1 = 199
Niter = 1000
frac_withheld = 0.1

features2drop = top_forest_imp[1:10]

AUC = pd.DataFrame(np.zeros((Niter,3+len(features2drop))),columns=np.hstack(['Full','No vol','Random',features2drop]))
AP = pd.DataFrame(np.zeros((Niter,3+len(features2drop))),columns=np.hstack(['Full','No vol','Random',features2drop]))

for i in tqdm(range(Niter)):
    
    df_g1s_balanced = rebalance_g1(df_g1s,Ng1)
    y_balanced = df_g1s_balanced['G1S_logistic']
    df_g1s_balanced = df_g1s_balanced.drop(columns='G1S_logistic')
    
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    _, _AUC,_AP = run_cross_validation(df_g1s_balanced,y_balanced,frac_withheld,forest)
    AUC.at[i,'Full'] = _AUC; AP.at[i,'Full'] = _AP
    
    forest_no_vol = RandomForestClassifier(n_estimators=100, random_state=42)
    _, _AUC,_AP = run_cross_validation(df_g1s_balanced.drop(columns='vol_sm'),y_balanced,frac_withheld,forest_no_vol)
    AUC.at[i,'No vol'] = _AUC; AP.at[i,'No vol'] = _AP
    
    df_random = pd.DataFrame(random.randn(*df_g1s_balanced.shape))
    random_model = RandomForestClassifier(n_estimators=100, random_state=42)
    _,_AUC,_AP = run_cross_validation(df_random,y_balanced,frac_withheld,random_model)
    AUC.at[i,'Random'] = _AUC; AP.at[i,'Random'] = _AP
    
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    for j in range(len(features2drop)):
        _, _AUC,_AP = run_cross_validation(df_g1s_balanced.drop(columns=features2drop[j]),y_balanced,frac_withheld,forest)
        AUC.at[i,features2drop[j]] = _AUC; AP.at[i,features2drop[j]] = _AP

I = AUC.mean().sort_values().index
sb.barplot(AUC[I])
plt.ylim([0.4,1]);plt.ylabel('Mean AUC');
plt.hlines(AUC.mean()['Full'],0,13)
plt.xticks(rotation=90);plt.tight_layout();

AUC_ = AUC.melt()
AUC_['Category'] = AUC_['variable']
AUC_.loc[(AUC_['variable'] != 'Full') & (AUC_['variable'] != 'Random') & (AUC_['variable'] != 'No vol'),'Category'] = 'No other'
AUC_.groupby('Category').mean()
# AUC_ = AUC_[AUC_['Category'] != 'Random']

sb.histplot(AUC_,x='value',hue='Category',bins = 15,stat='probability',element='poly',common_norm=False,fill=False)
plt.vlines(AUC_.groupby('Category').mean(),0,0.25)

#%% Single features; Random forest

Niter = 100

features2drop = top_forest_imp[1:5]
df_g1s['Intercept'] = 1

AUC = pd.DataFrame()
AP = pd.DataFrame()

for i in tqdm(range(Niter)):
    
    df_g1s_balanced = rebalance_g1(df_g1s,Ng1)
    y_balanced = df_g1s_balanced['G1S_logistic']
    df_g1s_balanced = df_g1s_balanced.drop(columns='G1S_logistic')
    
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    _, _AUC,_AP = run_cross_validation(df_g1s_balanced,y_balanced,frac_withheld,forest)
    AUC.at[i,'Full'] = _AUC; AP.at[i,'Full'] = _AP
    
    forest_no_vol = RandomForestClassifier(n_estimators=100, random_state=42)
    _, _AUC,_AP = run_cross_validation(df_g1s_balanced[['vol_sm','Intercept']],y_balanced,frac_withheld,forest_no_vol)
    AUC.at[i,'Only vol'] = _AUC; AP.at[i,'Only vol'] = _AP
    
    df_random = pd.DataFrame(random.randn(*df_g1s_balanced.shape))
    random_model = RandomForestClassifier(n_estimators=100, random_state=42)
    _,_AUC,_AP = run_cross_validation(df_random,y_balanced,frac_withheld,random_model)
    AUC.at[i,'Random'] = _AUC; AP.at[i,'Random'] = _AP
    
    for j in range(len(features2drop)):
        forest = RandomForestClassifier(n_estimators=100, random_state=42)
        _, _AUC,_AP = run_cross_validation(df_g1s_balanced[['Intercept',features2drop[j]]],y_balanced,frac_withheld,forest)
        AUC.at[i,f'Only {features2drop[j]}'] = _AUC; AP.at[i,f'Only {features2drop[j]}'] = _AP

AUC_ = pd.melt()
AUC_['Category'] = AUC_['variable']
AUC_.loc[(AUC_['variable']!= 'Full')&(AUC_['variable']!= 'Only vol')&(AUC_['variable']!= 'Random'),'Category'] = 'Other'

plt.figure()
sb.barplot(AUC_,y='value',x='variable',order=AUC.mean().sort_values(ascending=False).index)
plt.xticks(rotation=90);plt.tight_layout()

plt.figure()
sb.histplot(AUC_,bins=15,x='value',hue='Category',stat='probability',element='poly',common_norm=False,fill=False)
plt.vlines(AUC_.groupby('Category').mean(),0,0.25)

#%%

plt.figure()
hist_weights = np.ones(Niter)/Niter
plt.hist(AUC_full,histtype='step',weights=hist_weights)
plt.hist(AUC_vol,histtype='step',weights=hist_weights)
for i in range(6):
    plt.hist(AUC_single[i,:],histtype='step',weights=hist_weights)
plt.hist(AUC_random,histtype='step',weights=hist_weights)
plt.legend(np.hstack([ ['Full','Only volume'],other_features2test[:6],['Random'] ]))
plt.xlabel('AUC')

