#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:54:17 2019

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats
from os import path
from glob import glob

dirnames = ['/Users/xies/Box/Mouse/Skin/Fixed/11-06-2019 Skin Ecad488 EdU8h/']

conditions = ['WT','RB-KO']

vols = []
Cs = []
cell_type = []
names = []
region = []
phase = []
dapi = []
# Grab volume and DAPI
for dirname in dirnames:
    for cond in conditions:
        for fullname in glob(path.join(dirname,cond,'3/*.txt')):
            print fullname
            if path.splitext( path.splitext(fullname)[0])[1] == '.dapi':
                # Traverse up directory to see if Clover+ or Clover-
                cell_type.append( path.basename(path.dirname(fullname)) )
                Cs.append(cond)
                # Use data in filename to figure out bud# and phosRB status
                n = path.split( path.splitext(fullname)[0] )[-1]
                names.append(n)
                region.append(n.split('.')[0])
                if n.split('.')[2] == '-':
                    phase.append('G1')
                else:
                    phase.append('G2')
                # Add segmented area to get volume (um3)
                cell = pd.read_csv(fullname,delimiter='\t',index_col=0)
                vols.append(cell['Area'].sum())
                dapi.append(cell['RawIntDen'].sum())

df = pd.DataFrame()
df['Condition'] = Cs
df['Volume'] = vols
df['Cell'] = names
df['Type'] = cell_type
df['Region'] = region
df['DAPI'] = dapi
df['Phase'] = phase

g1 = df[df['Phase'] == 'G1']
wt = df[df['Condition'] == 'WT']
ko = df[df['Condition'] == 'RB-KO']

g1wt = wt[wt['Phase'] == 'G1']
g1ko = ko[ko['Phase'] == 'G1']


fig,ax = plt.subplots()
sb.boxplot(data = g1, x='Condition',y='Volume',notch=True, ax=ax)
sb.stripplot(data=g1,y='Volume',x='Condition',ax=ax,color='black',linewidth=1)

# Histogram
plt.hist(g1[g1['Condition'] == 'WT'].Volume,histtype='step')
plt.hist(g1[g1['Condition'] == 'RB-KO'].Volume,histtype='step')
plt.vlines(np.mean(g1[g1['Condition'] == 'WT'].Volume),0,12)
plt.vlines(np.mean(g1[g1['Condition'] == 'RB-KO'].Volume),0,12)
plt.xlabel('Nuclear volume (um3)')
plt.ylabel('Cell count')
plt.legend(['WT','RB-/-'])

sb.catplot(data=df,y='Volume',x='Condition',hue='Phase')



stats.ttest_ind(g1wt['Volume'],g1ko['Volume'])

