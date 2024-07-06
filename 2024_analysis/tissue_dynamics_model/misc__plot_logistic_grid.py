#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:38:41 2024

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb
from os import path

from sklearn import linear_model
from sklearn.metrics import roc_auc_score, log_loss

from tqdm import tqdm

df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_.csv',index_col=0)
df_['G1S_logistic'] = df_['Phase'] == 'SG2'
df_['random'] = np.random.randn(len(df_))

#%% Non balanced regressions

plt.figure()

fields2plot = ['Volume','Volume (sm)','Nuclear volume (sm)','Nuclear volume','Apical area'
               ,'Basal area','Mean neighbor FUCCI intensity','random']

scores = pd.DataFrame(index=fields2plot,columns=['AUC','Log loss'])

for i,field in enumerate(fields2plot):
    
    plt.subplot((len(fields2plot) -1 ) // 4 + 1, 4, i + 1)
    sb.regplot(df_,logistic=True, x=field, y='G1S_logistic', y_jitter=.1,scatter_kws={'alpha':0.05})

    X = df_[field].values.reshape(-1,1)
    y = df_['G1S_logistic']
    model = linear_model.LogisticRegression().fit(X,y)
    
    # Score the model
    score = roc_auc_score(y,model.predict(X))
    plt.title(f'AUC = {score:.2f}')
    
    scores.loc[field,'AUC'] = score
    
    score = log_loss(y,model.predict(X))
    plt.title(f'Log loss = {score:.2f}')
    
    scores.loc[field,'Log loss'] = score

#%% Balanced regressions

# plt.figure()
Niter = 1000

fields2plot = ['Volume','Volume (sm)','Nuclear volume (sm)','Nuclear volume','Apical area'
               ,'Basal area','Mean neighbor FUCCI intensity','random']

scores = pd.DataFrame(columns=fields2plot,index=range(Niter))
(_,g1),(_,sg2) = df_.groupby('G1S_logistic')

for i in tqdm(range(Niter)):
    
    df_sg2 = []
    for basalID,single_cell in sg2.groupby('basalID'):
        df_sg2.append(single_cell.sort_values('Frame').iloc[0])
    
    df_sg2 = pd.concat(df_sg2,axis=1,ignore_index=True).T
    df_g1 = g1.sample(n = 200)
        
    df_balanced = pd.concat((df_g1,df_sg2),ignore_index=True)
    
    for field in fields2plot:
    
        X = df_balanced[field].values.reshape(-1,1)
        y = df_balanced['G1S_logistic'].astype(bool)
        model = linear_model.LogisticRegression().fit(X,y)
        
        # Score the model
        score = roc_auc_score(y,model.predict(X))
        
        scores.loc[i,field] = score

#%%
sb.catplot(scores.melt(),x='variable',y='value',kind='box')

medians = scores.median()
g = plt.gca()
    
for xtick in g.get_xticks():
    g.text(xtick,scores.median()[xtick],f'{scores.median()[xtick]:.2f}', 
            horizontalalignment='center',size='x-small',color='w',weight='semibold')

plt.ylabel('AUC of 1000 randomly rebalanced dataset')
plt.xlabel('Single feature logistic regressor')
    
    
    