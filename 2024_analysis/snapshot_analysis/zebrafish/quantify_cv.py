#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:26:12 2024

@author: xies
"""

import numpy as np
import pandas as pd
from natsort import natsorted
from os import path
from glob import glob
from skimage import io, measure
from tqdm import tqdm

import xml.etree.ElementTree as ET

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/zebrafish_ditalia/osx_fucci_26hpp_11_4_17/'

#%%

birth = io.imread(path.join(dirname,'Position001_Mastodon/birth/birth_manual.tif'))

_df = []
for t in tqdm(range(birth.shape[0])):
    this_df = pd.DataFrame(measure.regionprops_table(birth[t,...],properties=['label','area']))
    this_df['Frame'] = t
    _df.append(this_df)
_df = pd.concat(_df)
_df = _df.rename(columns={'area':'Volume'})

df = pd.DataFrame(_df.groupby('label')['Volume'].mean())
df['Phase'] = 'Birth'

g1s = io.imread(path.join(dirname,'Position001_Mastodon/g1s/g1s_manual.tif'))

_df = []
for t in tqdm(range(g1s.shape[0])):
    this_df = pd.DataFrame(measure.regionprops_table(g1s[t,...],properties=['label','area']))
    this_df['Frame'] = t
    _df.append(this_df)
_df = pd.concat(_df)
_df = _df.rename(columns={'area':'Volume'})
_df = pd.DataFrame(_df.groupby('label')['Volume'].mean())
_df['Phase'] = 'G1S'

df = pd.concat((df,_df),ignore_index=True)

#%%