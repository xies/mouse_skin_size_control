#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:13:36 2023

@author: xies
"""


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb
import statsmodels.formula.api as smf

df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_.csv',index_col=0)
df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/Tissue model/df_g1s_final.csv',index_col=0)
df_g1s = df_g1s.drop(columns=['age','G1S_logistic'])

# # Categorical variable
regionnames = np.array(['R1','R2'])
df_g1s['region'] = regionnames[df_g1s['region'].values-1]

#Trim out G2 cells
df_g1s = df_g1s[df_g1s['time_g1s'] >= 0]

cellIDs = df_g1s['cellID']
df_g1s = df_g1s.drop(columns='cellID')


#%% Random effect grouped by cell (only intercept)

model = smf.mixedlm('time_g1s ~ ' + ' + '.join(df_g1s.columns.drop('time_g1s') + '+ vol_sm*sgr'), df_g1s, groups=cellIDs)
result = model.fit()
print(result.summary())

# R2 value
# https://stats.stackexchange.com/questions/578134/linear-mixed-model-r2-calculation-using-statsmodels
# https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/j.2041-210x.2012.00261.x

var_resid = result.scale
var_random_effect = float(result.cov_re.iloc[0])
var_fixed_effect = result.predict(df_g1s).var()
total_var = var_fixed_effect + var_random_effect + var_resid

marginal_r2 = var_fixed_effect / total_var
conditional_r2 = (var_fixed_effect + var_random_effect) / total_var

print(f'Marginal (fixed effect only) R2: {marginal_r2}')
print(f'Conditional (fixed effect + random effect) R2: {conditional_r2}')

#%%

print(f'----\nTop 10 significance param effect size: \n{result.params[result.pvalues.sort_values().index][0:10]}')
print(f'----\nTop 10 significance param conf interval (95%): \n{result.conf_int().loc[result.pvalues.sort_values().index].head(10)}')

#%%