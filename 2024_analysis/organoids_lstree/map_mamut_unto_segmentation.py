#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:58:19 2024

@author: xies
"""

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from os import path
from tqdm import tqdm

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'

df = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/filtered_tracks.csv'),index_col=0)

# Deaggregate into list of unique tracks
tracks = [t for _,t in df.groupby('TrackID')]

#%% Go through each track and 'filter' the segmentation

for t in tracks:
    