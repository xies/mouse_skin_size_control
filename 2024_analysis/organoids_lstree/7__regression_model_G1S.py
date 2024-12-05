#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 13:58:01 2024

@author: xies
"""

import numpy as np
import pandas as pd
from os import path
import seaborn as sb
from tqdm import tqdm
import matplotlib.pyplot as plt

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'
df5 = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_organoid_features_dynamic.csv'),index_col=0)
df5['organoidID'] = 5
df5 = df5[ (df5['cellID'] !=77) | (df5['cellID'] != 120)] #tetraploid/super weird
dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 2_2um/'
df2 = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_organoid_features_dynamic.csv'),index_col=0)
df2['organoidID'] = 2
df2 = df2[ (df2['cellID'] !=53) | (df2['cellID'] != 6)]
dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 31_2um/'
df31 = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_organoid_features_dynamic.csv'),index_col=0)
df31['organoidID'] = 31

df = pd.concat((df5,df2,df31),ignore_index=True)
df['organoidID_trackID'] = df['organoidID'].astype(str) + '_' + df['trackID'].astype(str)

# Derive some ratios
df['SA to vol ratio'] = df['Surface area'] / df['Nuclear volume']

#%% Preprocess
tracks = {trackID:t for trackID,t in df.groupby('organoidID_trackID')}

# First, drop everything but first G1/S frame
g1s_tracks = {}
for trackID,track in tracks.items():
    I = track['Auto phase']
    if I.sum() > 0:
        first_g1s_idx = np.where(I)[0][0]
        g1s_tracks[trackID] = track
        # g1s_tracks[trackID] = track.iloc[0:first_g1s_idx+1]

g1s = pd.concat(g1s_tracks, ignore_index=True)

#%%

from sklearn.preprocessing import scale
from sklearn.covariance import MinCovDet, EmpiricalCovariance
from numpy.linalg import eig

# @todo: neighbor interiority, orientation, neighbor orientation

feature_list = {'Nuclear volume (sm)':'nuc_vol',
                'Axial moment':'axial_moment',
                'Axial angle':'axial_angle',
                'Planar moment 1':'plan_moment_1',
                'Planar moment 2':'plan_moment_2',
                'Planar angle':'plan_angle',
                'Z':'z','Y':'y','X':'x',
                'Growth rates (sm)':'gr_sm',
                'Organoid interiority':'interior',
                'Mean curvature':'curvature',
                'Mean neighbor volume':'mean_neighb_vol',
                'Std neighbor volume':'std_neighb_vol',
                'Mean neighbor Cdt1 intensity':'mean_neighb_cdt1',
                'Mean neighbor Gem intensity':'mean_neighb_gem',
                'Local cell density':'local_density',
                'Change in local cell density':'delta_local_density',
                'Change in mean neighbor Cdt1':'delta_neighb_cdt1',
                'Change in mean neighbor Geminin':'delta_neighb_gem',

                'Change in mean neighbor size':'delta_neighb_vol',
                'Change in std neighbor size':'delta_std_neighb_vol',
                'Orientation':'orientation',

                # Drop these in real model -- redundant
                # 'Change in Cdt1':'delta_cdt1',
                # 'Change in Geminin':'delta_gem',
                # 'Mean Cdt1 intensity':'cdt1_int',
                # 'Mean Gem intensity':'gem_int',
                }

X = scale(g1s.loc[:,feature_list.keys()].dropna())

Cemp = EmpiricalCovariance().fit(X)
L,_ = eig(Cemp.covariance_)
print(f'Empirical cov: {L.max() / L.min():02f}')
Cmcv = MinCovDet().fit(X)
L,_ = eig(Cmcv.covariance_)
print(f'Min Cov Det cov: {L.max() / L.min():02f}')

#%% Instead of randomly subsampling, randomly sample one time piont from pre-G1 and post-G1 from each cell

def subsample_by_cell(df):

    subsampled = []
    for ID,this_cell in df.groupby('trackID'):

        pre_g1 = this_cell[ np.in1d(this_cell['Phase'],['Birth','Visible birth','G1']) ]
        post_g1 = this_cell[ ~np.in1d(this_cell['Phase'],['Birth','Visible birth','G1']) ]

        subsampled.append(post_g1.sample( min(5,len(post_g1))) )
        subsampled.append(pre_g1.sample(min(5,len(pre_g1))) )

    subsampled = pd.concat(subsampled)

    return subsampled

#%% Logistic regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from statsmodels.api import OLS
from scipy import stats

Niter = 100

coeffs = pd.DataFrame(columns=feature_list)
pvals = pd.DataFrame(columns=feature_list)
AP = pd.DataFrame()
AUC = pd.DataFrame()

Cmlr = np.zeros((Niter,2,2))

for i in tqdm(range(Niter)):

    balanced = subsample_by_cell(g1s)
    # balanced = g1s
    _df = balanced.loc[:,feature_list.keys()]
    _df['G1S_logistic'] = ~np.in1d(balanced['Phase'],['Birth','Visible birth','G1'])
    _df = _df.dropna()

    X = scale(_df.drop(columns='G1S_logistic'))
    y = _df['G1S_logistic']
    reg = OLS(y,X).fit()
    coeffs.loc[i,:] = reg.params.values.astype(float)
    pvals.loc[i,:] = reg.pvalues.values.astype(float)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = i)
    reg = LogisticRegression()
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    AP.loc[i,'data'] = average_precision_score(y_test,y_pred)
    AUC.loc[i,'data'] = roc_auc_score(y_test,y_pred)

    Cmlr[i,...] = confusion_matrix(y_test,y_pred)/len(y_test)

    X_rand = np.random.randn(*X_train.shape)
    reg = LogisticRegression()
    reg.fit(X_rand,y_train)
    y_pred = reg.predict(X_test)
    AP.loc[i,'random'] = average_precision_score(y_test,y_pred)
    AUC.loc[i,'random'] = roc_auc_score(y_test,y_pred)

# scores.plot.hist(bins=50)
# plt.tight_layout()

plt.figure()
plt.errorbar(x = coeffs.mean(),
              xerr = coeffs.std(),
              y = np.mean( -np.log10(pvals.values.astype(float)), axis = 0),
              yerr = np.std( -np.log10(pvals.values.astype(float)), axis = 0),
              fmt = 'o')
plt.hlines(y = -np.log10(.01), xmin=-.2, xmax=.4, color='r')
plt.xlabel('Coefficient')
plt.ylabel('-Log10 (P)')

plt.figure()

AP.plot.hist(); plt.xlabel('Average precision')
AUC.plot.hist(); plt.xlabel('AUC')

plt.figure()
sb.heatmap(Cmlr.mean(axis=0),annot=True)

#%%

from sklearn.inspection import permutation_importance

balanced = g1s
_df = balanced.loc[:,feature_list.keys()]
_df['G1S_logistic'] = ~np.in1d(balanced['Phase'],['Birth','Visible birth','G1'])
_df = _df.dropna()

X = scale(_df.drop(columns='G1S_logistic'))
y = _df['G1S_logistic']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = 42)
reg = LogisticRegression()
reg.fit(X_train,y_train)
r = permutation_importance(reg, X_test,y_test, n_repeats=100)

perm_imp = pd.DataFrame({'importance':r.importances_mean,
                         'std':r.importances_std},
                        index=_df.drop(columns='G1S_logistic').columns)
perm_imp.plot(kind='bar',y='importance',yerr='std');plt.tight_layout()
plt.ylabel('Permutation importance')

#%%

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

Niter = 50

score = np.zeros(Niter);
scores = pd.DataFrame()

for i in tqdm(range(Niter)):

    balanced = subsample_by_cell(g1s)
    _df = balanced.loc[:,feature_list.keys()]
    _df['G1S_logistic'] = ~np.in1d(balanced['Phase'],['Birth','Visible birth','G1'])
    _df = _df.dropna()

    X = scale(_df.drop(columns='G1S_logistic'))
    y = _df['G1S_logistic']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = i)
    forest = RandomForestClassifier()

    forest.fit(X_train,y_train)
    y_pred = forest.predict(X_test)

    scores.loc[i,'AP'] = average_precision_score(y_test,y_pred)
    scores.loc[i,'AUC'] = roc_auc_score(y_test,y_pred)

scores.plot.hist()

imp = pd.DataFrame({'importance':forest.feature_importances_},
                    index=feature_list)
imp.plot(kind='bar',y='importance'); plt.tight_layout()

#%%

from sklearn.inspection import permutation_importance

balanced = subsample_by_cell(g1s)
_df = balanced.loc[:,feature_list.keys()]
_df['G1S_logistic'] = ~np.in1d(balanced['Phase'],['Birth','Visible birth','G1'])
_df = _df.dropna()

X = scale(_df.drop(columns='G1S_logistic'))
y = _df['G1S_logistic']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = 42)
forest = RandomForestClassifier()
forest.fit(X_train,y_train)
r = permutation_importance(forest, X_test,y_test, n_repeats=50)

perm_imp = pd.DataFrame({'importance':r.importances_mean,
                         'std':r.importances_std},
                        index=_df.drop(columns='G1S_logistic').columns)
perm_imp.plot(kind='bar',y='importance',yerr='std');
plt.ylabel('Permutation importance'); plt.tight_layout()
