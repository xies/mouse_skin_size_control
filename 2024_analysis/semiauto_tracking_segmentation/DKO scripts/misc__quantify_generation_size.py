#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:33:43 2024

@author: xies
"""


import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from skimage import io, morphology, util, measure
from scipy.ndimage import convolve
import seaborn as sb
from os import path
from glob import glob
from tqdm import tqdm

import pickle as pkl

dirnames = ['/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/RBKO p107KO/M3 DOB 08-20-2023/11-07-2023 DKO ear (DOB 08-20-23, tam)/M3 p107homo Rbfl/Right ear/Post Ethanol/R1/',
           '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/RBKO p107KO/M3 DOB 08-20-2023/11-07-2023 DKO ear (DOB 08-20-23, tam)/M3 p107homo Rbfl/Left ear/Post tam/R1/']

dx = 0.2920097
dz = .75
# dx = 1

#%% Load parsed tracks, previous manual segtrack, additional segonly

first_gen = io.imread(path.join(dirnames[0],'Generation/first_gen_births.tif'))
second_gen = io.imread(path.join(dirnames[0],'Generation/second_gen_births.tif'))
dko = io.imread(path.join(dirnames[1],'Cell cycle/births.tif'))
df = []

for t in range(first_gen.shape[0]):
    _df = pd.DataFrame(measure.regionprops_table(first_gen[t,...],properties=['label','area']))
    _df['Generation'] = 1
    _df['Frame'] = t
    _df['Genotype'] = 'SKO'
    df.append(_df)
    _df = pd.DataFrame(measure.regionprops_table(dko[t,...],properties=['label','area']))
    _df['Generation'] = 2
    _df['Frame'] = t
    _df['Genotype'] = 'SKO'
    df.append(_df)
    _df = pd.DataFrame(measure.regionprops_table(dko[t,...],properties=['label','area']))
    _df['Frame'] = t
    _df['Genotype'] = 'DKO'
    df.append(_df)
    
df = pd.concat(df,ignore_index=True)
df['area'] *= dx**2*dz

sko = df[df['Genotype'] == 'SKO']
dko = df[df['Genotype'] == 'DKO']

#%%

sb.regplot(sko,x='Frame',y='area')

sb.lmplot(df,x='Frame',y='area',hue='Genotype')
plt.ylabel('Nulcear volume (fL)')
np.polyfit(df['Frame'],df['area'],1)

