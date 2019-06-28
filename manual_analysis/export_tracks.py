#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:30:01 2019

@author: xies@stanford
"""

import numpy as np
import pandas as pd
import seaborn as sb
import pickle
import os

####

# Load the region tracked cells
filename = '/data/Skin/W-R6/collated.pkl'
input_pkl = open(filename,'rb')
collated = pickle.load(input_pkl)
input_pkl.close()

# Grab the coordinates for each cell trace; construct directory hierarchy
basedir = os.path.split(filename)[0]

basedir_ = os.path.join(basedir,'tracked_cells')
if not os.path.exists(basedir_):
    os.mkdir( os.path.join(basedir,'tracked_cells'))

for c in collated:
    if len(c) < 2: # Don't bother with cells that are only a single time point
        continue
    newdir = os.path.join(basedir,'tracked_cells',str(np.unique(c['CellID'])[0]))
    if not os.path.exists(newdir):
        os.mkdir(newdir)
    
    T = np.array(c['Timeframe'])
    np.savetxt( os.path.join(newdir,'t.csv'), T, fmt='%d')
    X = np.array(c['PositionX'])
    np.savetxt( os.path.join(newdir,'x.csv'), X, fmt='%d')
    Y = np.array(c['PositionY'])
    np.savetxt( os.path.join(newdir,'y.csv'), Y, fmt='%d')
    