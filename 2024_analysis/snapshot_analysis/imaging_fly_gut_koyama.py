#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:58:07 2024

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from skimage import io,measure,util,segmentation
from os import path
from glob import glob
from natsort import natsort
from tqdm import tqdm

from SelectFromCollection import SelectFromCollection
from matplotlib.path import Path
from mathUtils import cvariation_ci


dirnames = []
dirnames.append('/Users/xies/OneDrive - Stanford/Shuyuan Hepatocytes/012821 Nusse Fucci hepatocytes HMGB1_SX')
dirnames.append('/Users/xies/OneDrive - Stanford/Shuyuan Hepatocytes/012621 Nusse Fucci hepatocytes RB-H2_SX')
# dirnames.append('/Users/xies/OneDrive - Stanford/Shuyuan Hepatocytes/012721 Nusse Fucci hepatocytes Rpb1CTD_SX')

#%%

df_all = []

for dirname in tqdm(dirnames):
    
    dapi = natsort.natsorted(glob(path.join(dirname,'*/Default/*channel000*z000.tif')))
    # dapi = list(map(io.imread,filelist))
    gem = natsort.natsorted(glob(path.join(dirname,'*/Default/*channel001*z000.tif')))
    # gem = list(map(io.imread,filelist))
    cdt = natsort.natsorted(glob(path.join(dirname,'*/Default/*channel002*z000.tif')))
    # cdt = list(map(io.imread,filelist))
    hmgb = natsort.natsorted(glob(path.join(dirname,'*/Default/*channel003*z000.tif')))
    # hmgb = list(map(io.imread,filelist))
    masks = natsort.natsorted(glob(path.join(dirname,'*/Default/*_masks.tif')))
    # masks = list(map(io.imread,filelist))
    
    im_tuples = list(zip(dapi,gem,cdt,hmgb,masks))
    
    for p,position in enumerate(im_tuples):
        
        mask = io.imread(position[4])
        mask_no_border = segmentation.clear_border(mask)
        dapi = io.imread(position[0]).astype(float)
        gem = io.imread(position[1]).astype(float)
        cdt = io.imread(position[2]).astype(float)
        hmgb = io.imread(position[3]).astype(float)
        
        threshold = dapi[mask == 0].mean()
        dapi = dapi - threshold; dapi[dapi < 0] = 0
        df = pd.DataFrame( measure.regionprops_table(mask_no_border,intensity_image = dapi,
                                       properties = ['label','axis_major_length','axis_minor_length','mean_intensity']))
        df = df.rename(columns={'mean_intensity':'DAPI'})
        df['Volume'] = df['axis_major_length']*df['axis_minor_length']*df['axis_minor_length']
        io.imsave(path.splitext(position[0])[0]+'_sub.tif',dapi.astype(np.uint16),check_contrast=False)
        
        threshold = gem[mask == 0].mean()
        gem = gem - threshold; gem[gem < 0] = 0
        df = pd.merge(df, pd.DataFrame( measure.regionprops_table(mask_no_border,intensity_image = gem,
                                       properties = ['label','mean_intensity'])))
        df = df.rename(columns={'mean_intensity':'Geminin'})
        io.imsave(path.splitext(position[1])[0]+'_sub.tif',gem.astype(np.uint16),check_contrast=False)
        
        threshold = cdt[mask == 0].mean()
        cdt = cdt - threshold; cdt[cdt < 0] = 0
        df = pd.merge(df, pd.DataFrame( measure.regionprops_table(mask_no_border,intensity_image = cdt,
                                       properties = ['label','mean_intensity'])))
        df = df.rename(columns={'mean_intensity':'Cdt1'})
        io.imsave(path.splitext(position[2])[0]+'_sub.tif',cdt.astype(np.uint16),check_contrast=False)
        
        threshold = hmgb[mask == 0].mean()
        hmgb = hmgb - threshold; hmgb[hmgb < 0] = 0
        df = pd.merge(df, pd.DataFrame( measure.regionprops_table(mask_no_border,intensity_image = hmgb,
                                       properties = ['label','mean_intensity'])))
        df = df.rename(columns={'mean_intensity':'HMGB'})
        io.imsave(path.splitext(position[3])[0]+'_sub.tif',hmgb.astype(np.uint16),check_contrast=False)
        
        df['Position'] = p
        df_all.append(df)
        
df = pd.concat(df_all,ignore_index=True)
df['Log_Cdt'] = np.log(df['Cdt1'])
df['Log_Gem'] = np.log(df['Geminin'])

#%%

pts = plt.scatter(df['Log_Cdt'],df['DAPI'],alpha=0.05)
plt.ylabel('DAPI')
plt.xlabel('Log_Cdt)')
# plt.xlim([0,25000])
# gate = roipoly()

selector = SelectFromCollection(plt.gca(), pts)

#%%

verts = np.array(selector.poly.verts)
x = verts[:,0]
y = verts[:,1]

p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(df['Volume'],df['DAPI'])])

diploid = df[I]

#%%

sb.lmplot(df,x='Volume',y='Log_Cdt', fit_reg=False, scatter_kws={'alpha':.1})
sb.lmplot(df,x='Volume',y='Log_Gem', fit_reg=False, scatter_kws={'alpha':.1})
sb.lmplot(diploid,x='Log_Cdt',y='Log_Gem',fit_reg=False, scatter_kws={'alpha':.1})

diploid['Geminin_high'] = True
diploid.loc[diploid['Log_Gem'] < 4,'Geminin_high'] = False

diploid['Cdt_high'] = True
diploid.loc[diploid['Log_Cdt'] < 2,'Cdt_high'] = False

sb.lmplot(diploid,x='Log_Cdt',y='Log_Gem',hue='Cdt_high', fit_reg=False, scatter_kws={'alpha':.1})
sb.lmplot(diploid,x='Log_Cdt',y='Log_Gem',hue='Geminin_high', fit_reg=False, scatter_kws={'alpha':.1})

#%%

g1 = diploid[diploid['Cdt_high'] & ~diploid['Geminin_high']]
sg2 = diploid[~diploid['Cdt_high'] & diploid['Geminin_high']]
g1s = diploid[diploid['Cdt_high'] & diploid['Geminin_high']]

def cv(X):
    return X.std()/X.mean()

print(cv(g1.Volume))
print(cv(g1s.Volume))
print(cv(sg2.Volume))




