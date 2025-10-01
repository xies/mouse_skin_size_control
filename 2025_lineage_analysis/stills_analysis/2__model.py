#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 16:48:47 2025

@author: xies
"""

# Core libraries
import numpy as np
from skimage import io, util, morphology, exposure, filters, measure
from scipy.spatial import Voronoi, Delaunay
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb

# Specific utils
from imageUtils import draw_labels_on_image, colorize_segmentation, normalize_exposure_by_axis
from trimesh import Trimesh
from trimesh.curvature import discrete_gaussian_curvature_measure, \
    discrete_mean_curvature_measure, sphere_ball_intersection
import pyvista as pv

# General utils
from os import path
import pickle as pkl
from tqdm import tqdm

dx = 0.3
dz = 0.5
Z_SHIFT = 10
KAPPA = 5 # microns

# for expansion
footprint = morphology.cube(3)

# Filenames
dirnames = {'R1':'/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/Shared/K10 paw/K10-R1',
            'R2':'/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/Shared/K10 paw/K10-R2'}

df = []
for name,dirname in dirnames.items():
    _df = pd.read_pickle(path.join(dirname,'Cropped/data_frame_aggregated.pkl'))
    _df['Region'] = name
    _df.index = name + '_' + _df.index.astype(str)
    df.append(_df)
    
df = pd.concat(df)

non_borders = df[~df.Border]

#%%

Niter = 100

from sklearn import ensemble, linear_model, pipeline
from sklearn import preprocessing, model_selection, decomposition
from sklearn import metrics

cols2drop = ['Border','Region']
cols2drop += [f for f in non_borders.columns if 'Y' in f or 'X' in f or 'Z' in f]
cols2drop += [f for f in non_borders.columns if 'K10' in f]
cols2drop += [f for f in non_borders.columns if 'Total' in f or 'cell coords' in f]

X = non_borders.drop(columns=cols2drop)
y = non_borders['Mean K10 intensity']

forest_scores = pd.DataFrame(index=range(Niter),columns=['R2_score'])
forest_imps = pd.DataFrame(index=range(Niter),columns=X.columns)
for i in tqdm(range(Niter)):
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y)
    
    p = pipeline.make_pipeline(preprocessing.StandardScaler(),
                               ensemble.RandomForestRegressor())
    p.fit(X_train,y_train)
    forest_imps.loc[i] = p[1].feature_importances_
    
    y_pred = p.predict(X_test)
    # plt.scatter(y_test,y_pred) 

    forest_scores.loc[i,'R2_score'] = metrics.r2_score(y_test,y_pred)
    
forest_imps.mean().sort_values().tail(20).plot.bar()
plt.tight_layout()

#%% XGboost

df = pd.read_pickle(path.join(dirname,'Cropped/data_frame_aggregated.pkl'))
non_borders = df[~df.Border]

Niter = 100

boost_scores = pd.DataFrame(index=range(Niter),columns=['R2_score'])
boost_imps = pd.DataFrame(index=range(Niter),columns=X.columns)
for i in tqdm(range(Niter)):
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y)
    
    p = pipeline.make_pipeline(preprocessing.StandardScaler(),
                               ensemble.GradientBoostingRegressor())
    p.fit(X_train,y_train)
    boost_imps.loc[i] = p[1].feature_importances_
    
    y_pred = p.predict(X_test)
    # plt.scatter(y_test,y_pred) 

    boost_scores.loc[i,'R2_score'] = metrics.r2_score(y_test,y_pred)
    
boost_imps.mean().sort_values().tail(20).plot.bar()
plt.tight_layout()

#%% Lin regression

Niter = 100

ols_scores = pd.DataFrame(index=range(Niter),columns=['R2_score'])
ols_imps = pd.DataFrame(index=range(Niter),columns=X.columns)

for i in tqdm(range(Niter)):
    
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y)
    p = pipeline.make_pipeline(preprocessing.StandardScaler(),
                               linear_model.LinearRegression())
    
    p.fit(X_train,y_train)
    ols_imps.loc[i] = p[1].coef_
    
    y_pred = p.predict(X_test)

    ols_scores.loc[i,'R2_score'] = metrics.r2_score(y_test,y_pred)

ols_imps.mean().abs().sort_values().tail(20).plot.bar()
plt.tight_layout()

#%% PCA regression

Niter = 100

# ols_scores = pd.DataFrame(index=range(Niter),columns=['R2_score'])
# ols_imps = pd.DataFrame(index=range(Niter),columns=X.columns)
pca_scores = pd.DataFrame(index=range(Niter),columns=['R2_score'])
pca_loadings = np.zeros((Niter, 30, len(X.columns)))
pca_coef = pd.DataFrame(index=range(Niter),columns=[f'PC{i}' for i in range(30)])

for i in tqdm(range(Niter)):
    
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y)
    p = pipeline.make_pipeline(preprocessing.StandardScaler(),
                               decomposition.PCA(n_components=30),
                               linear_model.LinearRegression())
    
    p.fit(X_train,y_train)
    pca_loadings[i,...] = p[1].components_
    pca_coef.loc[i] = p[2].coef_
    
    y_pred = p.predict(X_test)

    pca_scores.loc[i,'R2_score'] = metrics.r2_score(y_test,y_pred)

pca_loadings = pd.DataFrame(pca_loadings.mean(axis=0),index=[f'PC{i}' for i in range(30)],
                            columns=X.columns)

pca_coef.mean().abs().sort_values().tail(20).plot.bar()
plt.tight_layout()

#%%

forest_imps.mean().sort_values().tail(20).plot.bar()
plt.tight_layout()

#%%
from basicUtils import plot_bin_means

x='Subbasal collagen intensity'
y='Mean K10 intensity'
x='Basal area'

sb.regplot(non_borders,x=x,y=y,robust=True)

plot_bin_means(non_borders[x],non_borders[y],10)




