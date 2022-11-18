#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:40:03 2019

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats
from os import path
from glob import glob

dirnames = ['/Users/xies/Box/Mouse/Skin/Fixed/12-09-2019 Skin RB-het']

conditions = ['RB het male','WT male']

stains = ['DAPI EdU-8hr-594']

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
        for s in stains:
            for fullname in glob(path.join(dirname,cond,s,'*/*.txt')):
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
                        phase.append('SG2')
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

sb.catplot(data=df,y='Volume',x='Condition')

wt = df[df['Condition'] == 'WT male']
het = df[df['Condition'] == 'RB het male']

p = stats.ttest_ind(wt['Volume'],het['Volume'])

