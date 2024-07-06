#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:59:13 2023

@author: xies
"""

import numpy as np
from skimage import io

from glob import glob

dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/10-04-2023 R26CreER Rb-fl no tam ablation M5/M5 white DOB 4-25-23/R1'

#%%

T = 8

for t in range(T):
    
    