#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:24:43 2019

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pickle as pkl

#Load df from pickle
r1 = pd.read_pickle('/Users/xies/Box/Mouse/Skin/W-R1/tracked_cells/dataframe.pkl')
r2 = pd.read_pickle('/Users/xies/Box/Mouse/Skin/W-R2/tracked_cells/dataframe.pkl')
r5 = pd.read_pickle('/Users/xies/Box/Mouse/Skin/W-R5/tracked_cells/dataframe.pkl')
df = pd.concat((r1,r2,r5))

df = df[~df.Mitosis]
Ncells = len(df)

# Load growth curves from pickle
with open('/Users/xies/Box/Mouse/Skin/W-R1/tracked_cells/collated_manual.pkl','rb') as f:
    c1 = pkl.load(f)
with open('/Users/xies/Box/Mouse/Skin/W-R2/tracked_cells/collated_manual.pkl','rb') as f:
    c2 = pkl.load(f)
with open('/Users/xies/Box/Mouse/Skin/W-R5/tracked_cells/collated_manual.pkl','rb') as f:
    c5 = pkl.load(f)
collated = c1 + c2 + c5

ucellIDs = np.array([c.iloc[0].CellID for c in collated])


# Plot daughter growth curves
has_daughter = df[~np.isnan(df['Daughter a volume'])]
plt.hist(nonans(df['Daughter ratio']))
plt.xlabel('Daughter volume ratio')
plt.ylabel('Frequency')

for i in has_daughter.CellID.values:
    I = np.where(ucellIDs == i)[0][0]
    c = collated[I]
    mainC = c[c['Daughter'] == 'None']
    t = (mainC.Frame - mainC.iloc[0].Frame)*12
    daughters = c[c['Daughter'] != 'None']
    plt.plot(t,mainC.Volume,'b')
    if daughters.iloc[0].Frame == 2:
        print i
    plt.plot([t.iloc[-1], t.iloc[-1] + 6],
             [mainC.iloc[-1].Volume,daughters.Volume.sum()],
             marker='o',linestyle='dashed',color='r')
    plt.xlabel('Time since birth (hr)')
    plt.ylabel('Cell volume')
    