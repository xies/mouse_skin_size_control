#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:17:47 2024

@author: xies
"""

from skimage import io, measure
from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/RBKO p107KO/M3 DOB 08-20-2023/11-07-2023 DKO ear [DOB 08-20-23, tam]/M3 p107homo Rbfl/Right ear/Post Ethanol/R1/'
dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/RBKO p107KO/M3 DOB 08-20-2023/11-07-2023 DKO ear [DOB 08-20-23, tam]/M3 p107homo Rbfl/Left ear/Post tam/R1/'

R = io.imread(path.join(dirname,'master_stack/R.tif'))
tracking = io.imread(path.join(dirname,'manual_tracking/curated_clahe.tif'))

#%%

df = []
for t in range(tracking.shape[0]):
    
    _df = pd.DataFrame(measure.regionprops_table(tracking[t,...], intensity_image = R[t,...],
                                   properties=['label','mean_intensity','area']))
    _df['Frame'] = t
    df.append(_df)
    
df = pd.concat(df)

#%%

collated = [c for _,c in df.groupby('label')]

c = collated[0]

plt.plot(c.Frame,c['mean_intensity'])
plt.xlabel('Frame')
plt.xlabel('FUCCI-G1 intensity (a.u.)')

#%%

lengths = np.array([len(c) for c in collated])
sortedI = lengths.argsort()
fucci_mat = np.ones((len(collated), int(lengths.max()))) * np.nan

for i in range(len(collated)):
    c = collated[sortedI[i]]
    fucci_mat[i,: len(c)] = c.mean_intensity
    
XX,YY = np.meshgrid( np.arange(0,lengths.max()) * 12, range(len(collated)))
plt.pcolor(XX,YY,fucci_mat, cmap = 'hot')
plt.xlabel('Cell age (h)')
plt.ylabel('CellID')

plt.colorbar()