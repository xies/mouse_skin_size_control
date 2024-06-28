#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:41:49 2024

@author: xies
"""

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from os import path

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn import linear_model

from tqdm import tqdm

#%%

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
with open(path.join(dirname,'basal_no_daughters.pkl'),'rb') as f:
    c1 = pkl.load(f)
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'
with open(path.join(dirname,'basal_no_daughters.pkl'),'rb') as f:
    c2 = pkl.load(f)
    
collated = list(c1.values()) + list(c2.values())

#%%
df = pd.DataFrame()
df['Birth nuclear volume'] = np.array([c.iloc[0]['Nuclear volume (sm)'] for c in collated])
df['Birth volume'] = np.array([c.iloc[0]['Volume (sm)'] for c in collated])
df['Exponential growth rate'] = np.array([c.iloc[0]['Exponential growth rate'] for c in collated])

g1_duration = np.zeros(len(collated))
for i,c in enumerate(collated):
    I = np.where(c['Phase'] == 'SG2')[0]
    if len(I)>0:
        g1_duration[i] = c.iloc[I[0]].Age

df['G1 duration'] = g1_duration

#%%

X = df.drop(columns='G1 duration')
y = df['G1 duration']

X = scale(X)

#%% Linear model: test split ratios

split_ratios = np.linspace(0.5,0.95,10)
Niter = 100
r2 = np.zeros((len(split_ratios),Niter))

for j,th in tqdm(enumerate(split_ratios)):
    for i in range(100):
        
        X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=th)
        lin_model = linear_model.LinearRegression()
        lin_model.fit(X_train,y_train)
        y_pred = lin_model.predict(X_test)
        r2[j,i] = r2_score(y_test,y_pred)
        
plt.errorbar(split_ratios,r2.mean(axis=1),r2.std(axis=1))
plt.xlabel('Training set ratio'); plt.ylabel('R2 score')

#%%

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=th)
lin_model = linear_model.LinearRegression().fit(X,y)
perm_imp = permutation_importance(lin_model,X,y)




