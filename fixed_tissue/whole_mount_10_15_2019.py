#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 00:13:23 2019

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from os import path
from glob import glob
from scipy import stats

dirname = '/Users/xies/Box/Mouse/Skin/Fixed/10-11-2019 Skin/'
conditions = ['WT','RB-KO']
stain = 'DAPI Ecad-488'

vols = []
Cs = []
cell_type = []
names = []
organoid = []
phosRB = []
dapi = []
# Grab volume and DAPI
filelist = []
for cond in conditions:
    for fullname in glob(path.join(dirname,cond,stain,'*/*.txt')):
        print fullname
        if path.splitext(fullname)[1] == '.txt':
            # Traverse up directory to see if Clover+ or Clover-
            cell_type.append( path.basename(path.dirname(fullname)) )
            Cs.append(cond)
            # Use data in filename to figure out bud# and phosRB status
            n = path.split( path.splitext(fullname)[0] )[-1]
            names.append(n)
            organoid.append(n.split('.')[0])
            phosRB.append(n.split('.')[-1])
            # Add segmented area to get volume (um3)
            cell = pd.read_csv(fullname,delimiter='\t',index_col=0)
            vols.append(cell['Area'].sum())
            dapi.append(cell['IntDen'].sum())

df = pd.DataFrame()
df['Condition'] = Cs
df['Volume'] = vols
df['Cell'] = names
df['Type'] = cell_type
df['Organoid'] = organoid
df['DAPI'] = dapi
df['Condition Type'] = df['Condition'] + df['Type']

