import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path

import napari
import numpy as np
from napari.utils.notebook_display import nbscreenshot
from tqdm import tqdm
from rich.pretty import pprint
from natsort import natsorted
from os import path
from glob import glob
from skimage import io

from ultrack import segment, link, solve, to_tracks_layer, tracks_to_zarr
from ultrack.utils import estimate_parameters_from_labels, labels_to_edges
from ultrack.config import MainConfig

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

viewer = napari.Viewer()
nuclei = np.stack(map(io.imread,natsorted(glob(path.join(dirname,'3d_nuc_seg/cellpose_cleaned_manual/t[0-3].tif')))))[:,:,:200,:200]
viewer = napari.Viewer()
viewer.add_image(nuclei)

cellpose_labels = viewer.layers['nuclei'].data

detection = []; edges = []
for l in cellpose_labels:
    det,e = labels_to_edges(cellpose_labels,sigma=4.0)
    detection.append(det); edges.append(e)
detection = np.stack(detection)
edges= np.stack(edges)

##
config = MainConfig()

# Segmentation parameters
config.segmentation_config.min_area = 500
config.segmentation_config.max_area = 5000
config.segmentation_config.n_workers = 8
# Linking parameters
config.linking_config.max_distance = 25 #in pixels
config.linking_config.n_workers = 2
# Tracking
config.tracking_config.appear_weight = -0.001
config.tracking_config.disappear_weight = -0.01
config.tracking_config.division_weight = -0.01

config.tracking_config.power = 4 # Penalty for topologic changes
config.tracking_config.bias = -0.001
config.tracking_config.solution_gap = 0.0

segment( detection = detection,
        edge = edges,
        config = config,
        overwrite=True,
        )

link(config=config,
     overwrite=True
     )

solve(config=config,
      overwrite=True
      )

tracks_df, graph = to_tracks_layer(config.data_config)
labels = tracks_to_zarr(config.data_config, tracks_df)


viewer.add_tracks(tracks_df[["track_id", "t", "y", "x"]].values, graph=graph)
viewer.add_labels(labels)

