#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 19:17:55 2024

@author: xies
"""


import pandas as pd
import numpy as np
from os import path
import matplotlib.pyplot as plt
import seaborn as sb
from basicUtils import plot_bin_means

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'
df5 = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_features.csv'),index_col=0)
df5['organoidID'] = 5
df5 = df5[ (df5['cellID'] !=77) | (df5['cellID'] != 120)]
dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 2_2um/'
df2 = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_features.csv'),index_col=0)
df2['organoidID'] = 2
df2 = df2[ (df2['cellID'] !=53) | (df2['cellID'] != 6)]
dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 31_2um/'
df31 = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_features.csv'),index_col=0)
df31['organoidID'] = 31

regen = pd.concat((df5,df2,df31),ignore_index=True)
regen['organoidID_trackID'] = regen['organoidID'].astype(str) + '_' + regen['trackID'].astype(str)
regen['Cell type'] = 'Regenerative'

regen['Specific GR (sm)'] = regen['Growth rates (sm)'] * 60 / regen['Nuclear volume (sm)'] # per hour

#%%

dirname = '/Users/xies/OneDrive - Stanford/In vitro/mIOs/Light sheet movies/20200303_194709_09'
homeo = pd.read_csv(path.join(dirname,'growth_rates.csv'),index_col=0)
homeo = homeo.rename(columns={'Volume sm':'Nuclear volume (sm)','Type':'Cell type',
                              'Volume':'Nuclear volume','Specific GR sm':'Specific GR (sm)'})


#%% Concatenate

regen_g1 = regen[ (regen['Phase'] == 'G1') | (regen['Phase'] == 'G1S') ]

fields2concat = ['Cell type','Nuclear volume','Specific GR (sm)']
df = pd.concat((regen_g1[fields2concat],homeo[fields2concat]),ignore_index=True)

# sb.lmplot(df,x='Nuclear volume',y='Specific GR (sm)',hue='Cell type')


colors = {'Regenerative':'b','TA':'m','Stem cell':'g'}

plt.figure()
names = []
for name, celltype in df.groupby('Cell type'):
    # plt.scatter(celltype['Nuclear volume'],celltype['Specific GR (sm)'], color=colors[name],alpha=0.1)
    plot_bin_means(celltype['Nuclear volume'],celltype['Specific GR (sm)'],bin_edges=15,minimum_n=25,
                   color=colors[name], style='fill')
    names.append(name)

plt.xlim([100,275])
plt.ylim([-0.05,0.3])
plt.xlabel('Nuclear volume (fL)')
plt.ylabel('Specific growth rate (h-1)')
plt.legend(names)

plt.figure()
sb.catplot(df,x='Cell type',y='Specific GR (sm)', kind='box')
