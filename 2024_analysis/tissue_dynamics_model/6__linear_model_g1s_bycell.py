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
import seaborn as sb
from os import path

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn import linear_model
from basicUtils import jitter

from numpy import random
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
df['Birth NC ratio'] = df['Birth nuclear volume'] / df['Birth volume']
df['Exponential growth rate'] = np.array([c.iloc[0]['Exponential growth rate'] for c in collated])
df['Exponential nuc growth rate'] = np.array([c.iloc[0]['Exponential nuc growth rate'] for c in collated])
df['Exponential growth rate G1 only'] = np.exp(np.array([c.iloc[0]['Exponential growth rate G1 only'] for c in collated]))
df['Exponential nuc growth rate G1 only'] = np.exp(np.array([c.iloc[0]['Exponential nuc growth rate G1 only'] for c in collated]))

g1_duration = np.ones(len(collated))*np.nan
for i,c in enumerate(collated):
    I = np.where(c['Phase'] == 'SG2')[0]
    if len(I)>0:
        g1_duration[i] = c.iloc[I[0]].Age

df['G1 duration'] = g1_duration
df = df.dropna()

#%% Linear regression 

endo_var_names = ['Birth volume','Birth nuclear volume','Exponential growth rate']
var_names_str = ' + '.join(endo_var_names)
X = df[endo_var_names].copy()
y = df['G1 duration']

X_ = scale(X)

#% Linear model for in sample R2 score

lin_model = linear_model.LinearRegression()
lin_model.fit(X_,y)
y_pred = lin_model.predict(X_)
R2 = r2_score(y,y_pred)

plt.figure()
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r')
plt.plot([y.min(),y.max()],[y.min()-6,y.max()-6],'r--')
plt.plot([y.min(),y.max()],[y.min()+6,y.max()+6],'r--')
plt.scatter(jitter(y,sigma=2.5),y_pred, alpha=0.5,s=50)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('Measured G1 duration')
plt.ylabel('Predicted G1 duration')
plt.title(f'Linear regression using: {var_names_str}; R2={R2:.2f}; maxR2=0.88')

#%% Renormalize R2 given sampling rate of empirical data

T = random.normal(size=10000)*21 + 56
T = T[T>0]
bins = np.arange(0,1000,12)
which_bin = np.digitize(T,bins)
Tmeasured = bins[which_bin-1]
plt.scatter(T,Tmeasured)
r2_score(T,Tmeasured)

#%% Linear model: test split ratios and report out of sample R2 scores

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
plt.title('Out of model R2 score')

#%% Permutation importance

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=th)
lin_model = linear_model.LinearRegression().fit(X,y)
perm_imp = permutation_importance(lin_model,X,y,n_repeats=100)
results = pd.DataFrame(perm_imp['importances'].T,columns=df.columns.drop('G1 duration').values,index=range(100))

sb.barplot(results)
plt.ylabel('Feature importance')



