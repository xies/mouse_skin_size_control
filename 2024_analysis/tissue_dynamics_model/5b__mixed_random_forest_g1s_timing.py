#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:53:13 2023

@author: xies
"""


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb

df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_.csv',index_col=0)
df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_g1s.csv',index_col=0)
df_g1s = df_g1s.drop(columns=['age','G1S_logistic'])

# # Categorical variable
regionnames = np.array(['R1','R2'])
df_g1s['region'] = regionnames[df_g1s['region'].values-1]

#Trim out G2 cells
df_g1s = df_g1s[df_g1s['time_g1s'] >= 0]

cellIDs = df_g1s['cellID']
df_g1s = df_g1s.drop(columns=['region'])

y = df_g1s['time_g1s']
X = df_g1s.drop(columns='time_g1s')

#%% Random effect grouped by cell (only intercept)

from merf import MERF
from sklearn.model_selection import train_test_split
from merf.viz import plot_merf_training_stats

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
clusters_train = X_train['cellID']
X_train = X_train.drop(columns='cellID')
clusters_test = X_test['cellID']
X_test = X_test.drop(columns='cellID')

Ntrain = len(y_train)
Nclusters = len(np.unique(clusters_test))
# Need to construct Z_train, i.e. training

model = MERF()
model = model.fit(X_train, np.ones((Ntrain,1)), clusters_train, y_train)
plot_merf_training_stats(model, num_clusters_to_plot=10)

y_hat = model.predict(X_test, np.ones((len(y_test),1)), clusters_test)
R2 = np.corrcoef(y_hat,y_test)[0,1]**2
print(f'R2 (known) = {R2}')

#%% Example
# https://towardsdatascience.com/mixed-effects-random-forests-6ecbb85cb177

from merf.utils import MERFDataGenerator
from merf.merf import MERF


dgm = MERFDataGenerator(m=.6, sigma_b=np.sqrt(4.5), sigma_e=1)

num_clusters_each_size = 20
train_sizes = [1, 3, 5, 7, 9]
known_sizes = [9, 27, 45, 63, 81]
new_sizes = [10, 30, 50, 70, 90]

train_cluster_sizes = MERFDataGenerator.create_cluster_sizes_array(train_sizes, num_clusters_each_size)
known_cluster_sizes = MERFDataGenerator.create_cluster_sizes_array(known_sizes, num_clusters_each_size)
new_cluster_sizes = MERFDataGenerator.create_cluster_sizes_array(new_sizes, num_clusters_each_size)

train, test_known, test_new, training_cluster_ids, ptev, prev = dgm.generate_split_samples(train_cluster_sizes, known_cluster_sizes, new_cluster_sizes)

X_train = train[['X_0', 'X_1', 'X_2']]
Z_train = train[['Z']]
clusters_train = train['cluster']
y_train = train['y']

val = pd.concat([test_known, test_new])
X_val = val[['X_0', 'X_1', 'X_2']]
Z_val = val[['Z']]
clusters_val = val['cluster']
y_val = val['y']

mrf = MERF(max_iterations=5)
mrf.fit(X_train, Z_train, clusters_train, y_train)

plot_merf_training_stats(mrf, num_clusters_to_plot=10)