#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 00:06:07 2023

@author: xies
"""

import numpy as np
from numpy import random

#%%

MAX_FRAME = 20

Niter = 250

frames = []
age = []
V = []

special = np.zeros(Niter,dtype=bool)

for i in range(Niter):
    
    t = np.arange(0,MAX_FRAME)
    k = 0.6 * (1 + 0.2*random.randn())
    bsize = 100 * (1 + 0.25*random.rand())
 
    birth_frame = random.randint(1,high=MAX_FRAME-1)

    y = bsize + np.exp(k*(t-birth_frame))

    y[t < birth_frame] = np.nan
    y[(y - bsize) > 100] = np.nan
    
    if not np.isnan(y[-1]):
        continue
    
    t = t[~np.isnan(y)]
    y = y[~np.isnan(y)]
    
    
    # ~10% White noise
    y = y * (1 + 0.1*random.rand(len(y)))
    
    
    V.append(y)
    frames.append(t)
    age.append(t-t[0])
    plt.plot(t,y)
    

#%%

bad_frame = [2,9]
magnitudes =[-20,+20]
for i,(t,y) in enumerate(zip(frames,V)):
    
    for bad,mag in zip(bad_frame,magnitudes):
        # Add noise to specific frame
        if 2 in t:
            y[t == bad] += mag * (1+0.1*random.rand())
    
    plt.plot(t,y)
    
    
#%%

birth_sizes = np.zeros(len(V))
div_sizes = np.zeros(len(V))
cyc_lengths = np.zeros(len(V))
birth_frames = np.zeros(len(V))

for i,(t,y) in enumerate(zip(frames,V)):

    birth_sizes[i] = y[0]
    div_sizes[i] = y[-1]
    birth_frames[i] = t[0]
    cyc_lengths[i] = t[-1]


plt.figure()
plt.scatter(np.log(birth_sizes), cyc_lengths)
p = np.polyfit(np.log(birth_sizes), cyc_lengths,1)
print(f'Length slope = {p[0]}')

plt.figure()
plt.scatter(birth_sizes, div_sizes-birth_sizes)

p = np.polyfit(birth_sizes, div_sizes-birth_sizes,1)

print(f'Growth slope = {p[0]}')

