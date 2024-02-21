#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:25:31 2024

@author: xies
"""

import numpy as np
import pandas as pd
from os import path
from skimage import io,draw
from tqdm import tqdm

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
df_all = pd.read_csv(path.join(dirname,'tissue_dataframe.csv'),index_col=0)

s = 20

im = io.imread(path.join(dirname,f'Image flattening/flat_cyto_seg_manual/t0.tif'))
XX,__ = im.shape

for t,df in tqdm(df_all.groupby('Frame')):
    
    df = df.dropna(subset=['Collagen orientation','Basal orientation','Nuclear volume'])
    
    Y = df['Y-pixels']
    X = df['X-pixels']
    
    scale = df['Collagen fibrousness'] * s
    
    theta = df['Collagen orientation']
    U = -np.sin(np.deg2rad(theta))* scale
    V = np.cos(np.deg2rad(theta))* scale
    
    # Align onto midpoint of vector 
    U0 = -U/2
    V0 = -V/2
    Uf = U/2
    Vf = V/2
    
    LY0 = np.round(U0 + Y).astype(int).values
    LX0 = np.round(V0 + X).astype(int).values
    LYf = np.round(Uf + Y).astype(int).values
    LXf = np.round(Vf + X).astype(int).values
    
    arrows = np.zeros_like(im)
    for i in range(len(df)):
        RR,CC = draw.line(LY0[i],LX0[i],LYf[i],LXf[i])
        
        # Enforce boundaries
        RR[RR>=XX] = XX-1; CC[CC>=XX] = XX-1
        RR[RR<0] = 0; CC[CC<0] = 0
        
        arrows[RR,CC] = df.iloc[i].CellposeID
        
    # break
    
    io.imsave(path.join(dirname,f'Image flattening/collagen_orientation/t{t}_as_line.tif'),arrows)
    