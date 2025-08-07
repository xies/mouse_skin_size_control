#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 11:47:54 2025

@author: xies
"""



import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from skimage import io, measure
import seaborn as sb
from os import path
from glob import glob
from tqdm import tqdm
from natsort import natsorted

import pickle as pkl
from re import match

from basicUtils import *

dirnames = {}

dirnames['DKO_R1'] = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/RBKO p107KO/M3 DOB 08-20-2023/11-07-2023 DKO ear (DOB 08-20-23, tam)/M3 p107homo Rbfl/Left ear/Post tam/R1/'
dirnames['DKO_R2'] = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/RBKO p107KO/M3 DOB 08-20-2023/11-07-2023 DKO ear (DOB 08-20-23, tam)/M3 p107homo Rbfl/Left ear/Post tam/R2/'
dirnames['DKO_R5'] = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/RBKO p107KO/M3 DOB 08-20-2023/11-07-2023 DKO ear (DOB 08-20-23, tam)/M3 p107homo Rbfl/Left ear/Post tam/R5/'
dirnames['SKO_R1'] = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/RBKO p107KO/M3 DOB 08-20-2023/11-07-2023 DKO ear (DOB 08-20-23, tam)/M3 p107homo Rbfl/Right ear/Post Ethanol/R1/'
dirnames['SKO_R2'] = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/RBKO p107KO/M3 DOB 08-20-2023/11-07-2023 DKO ear (DOB 08-20-23, tam)/M3 p107homo Rbfl/Right ear/Post Ethanol/R2/'
dirnames['SKO_R3'] = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/RBKO p107KO/M3 DOB 08-20-2023/11-07-2023 DKO ear (DOB 08-20-23, tam)/M3 p107homo Rbfl/Right ear/Post Ethanol/R3/'


#%%

df = pd.DataFrame()

index = 0
for region,dirname in dirnames.items():
    filelist = natsorted(glob(path.join(dirname,'dead_cells/t*.csv')))
    for f in filelist:
        index += 1
        t = int(match('t(\d+).csv',path.split(f)[1]).groups()[0])
    
        df.loc[index,'Count'] = len(pd.read_csv(f))
        df.loc[index,'Region'] = region
        df.loc[index,'Genotype'] = region.split('_')[0]
        df.loc[index,'Time'] = t - 10

df.to_excel('/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/RBKO p107KO/M3 DOB 08-20-2023/11-07-2023 DKO ear (DOB 08-20-23, tam)/M3 p107homo Rbfl/dead_cell_summary.xlsx')
