#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:11:48 2024

@author: xies
"""

import numpy as np
import pandas as pd
import seaborn as sb
from os import path

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Confocal/01-17-2024 M3 p107-homo RB1-fl DNA damage stains test/'

regions = ['tam_53BP1_1','ethanol_53BP1_1','tam_53BP1_2','ethanol_53BP1_2']

#%%

df = []

for region in regions:
    
    cell_counts = pd.read_csv(path.join(dirname,region,'quantification/cells2count.csv'))
    Ncells = len(cell_counts)
    _df = pd.read_excel(path.join(dirname,region,'quantification/foci_per_cell.xlsx'))
    
    counts = np.zeros(Ncells)
    counts[:len(_df)] = _df['Counts per cell']
    
    _df = pd.DataFrame(counts,columns=['53BP1 foci'])
    geno = region.split('_')[0]
    _df['Region'] = region
    _df['Genotype'] = geno
    df.append(_df)

df = pd.concat(df,ignore_index=True)

from basicUtils import ttest_from_groupby

_,P  = ttest_from_groupby(df,'Genotype','53BP1 foci')
print(f'T-test between genotype: {P}')

#%%

# sb.histplot(df,hue='Genotype',x='53BP1 foci',cumulative=True,element='step', fill=False)
df_jittered = df.copy()
df_jittered['53BP1 foci'] = df['53BP1 foci'] + np.random.randn(len(df)) * .1
sb.stripplot(df_jittered,x='Genotype',y='53BP1 foci')



# import seaborn.objects as so
# (
#     so.Plot(df, "Genotype", "53BP1 foci")
#     .add(so.Dots(), so.Jitter(x = .3,y=.1))
# )

