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
from natsort import natsorted

from skimage import io
from os import path
from glob import glob

import napari
import numpy as np
from napari.utils.notebook_display import nbscreenshot
from tqdm import tqdm
from rich.pretty import pprint

from ultrack import MainConfig, track, to_tracks_layer, tracks_to_zarr
from ultrack.utils.array import array_apply, create_zarr
from ultrack.imgproc.segmentation import reconstruction_by_dilation
from ultrack.imgproc.plantseg import PlantSeg
from ultrack.utils.cuda import import_module, to_cpu, on_gpu

#%%

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

nuclei = io.imread(path.join(dirname,'Cropped_images/B.tif'))
membranes = io.imread(path.join(dirname,'Cropped_images/G.tif'))

detection = np.stack(map(io.imread,natsorted(glob(path.join(dirname,'3d_nuc_seg/cellpose_cleaned_manual/t*.tif')))))
boundaries = np.stack(map(io.imread,natsorted(glob(path.join(dirname,'3d_cyto_seg/3d_cyto_manual/t*_cleaned.tif')))))

viewer = napari.Viewer()
viewer.add_image(nuclei,colormap='blue')
viewer.add_image(membranes,blending='additive')
viewer.add_labels(detection)
viewer.add_labels(boundaries)

scale = [1,.25,.25]
for l in viewer.layers:
    l.scale = [1,.25,.25]

#%% CONFIG

cfg = MainConfig()

cfg.data_config.n_workers = 8

cfg.segmentation_config.n_workers = 6
cfg.segmentation_config.min_area = 2000
cfg.segmentation_config.max_area = 20000
cfg.segmentation_config.min_frontier = 0.35

cfg.linking_config.n_workers = 16
cfg.linking_config.max_neighbors = 5
cfg.linking_config.max_distance = 3.0  # microns
cfg.linking_config.distance_weight = 0.01

cfg.tracking_config.appear_weight = -1.0
cfg.tracking_config.disappear_weight = -1.0
cfg.tracking_config.division_weight -0.1

pprint(cfg)

#%% Start tracking

track(
    cfg,
    detection=detection,
    edges=boundaries,
    overwrite=True,
    scale=scale,
)
