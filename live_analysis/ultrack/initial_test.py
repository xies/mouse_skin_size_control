#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:06:32 2023

@author: xies
"""

# stardist / tensorflow env variables setup
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path

from os import path
from glob import glob

import napari
import numpy as np
from napari.utils.notebook_display import nbscreenshot
from tqdm import tqdm
from rich.pretty import pprint

from ultrack import segment, link, solve, to_tracks_layer, tracks_to_zarr
from ultrack.utils import estimate_parameters_from_labels, labels_to_edges
from ultrack.config import MainConfig

#%%

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/3d_nuc_seg/cellpose_cleaned_manual'

viewer = napari.viewer()
viewer.open(sorted())