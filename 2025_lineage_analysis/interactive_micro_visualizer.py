import napari
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from magicgui import magicgui
from napari.layers import Labels
from os import path
import pickle as pkl
import numpy as np
from glob import glob
from natsort import natsorted

import sched, time
from imageUtils import trim_multimasks_to_shared_bounding_box, pad_image_to_size_centered
from visualization import extract_nuc_and_cell_and_microenvironment_mask_from_idx,get_nuc_and_cell_and_microenvironment_movie


@napari.Viewer.bind_key('c')
def cycle_colormap(viewer):
    # grab active layer
    current_layer = viewer.layers.selection.active
    # if it's a labels layer, cycle the colormap
    if isinstance(current_layer,napari.layers.labels.labels.Labels):
        if current_layer.show_selected_label:
            REISOLATE=True
            current_layer.show_selected_label = False
        else:
            REISOLATE = False
        viewer.status = 'Cycling random colormap'
        current_layer.new_colormap()
        if REISOLATE:
            current_layer.show_selected_label = True

@napari.Viewer.bind_key('i')
def isolate_label(viwer):
    # grab the active layer
    current_layer = viewer.layers.selection.active
    if isinstance(current_layer,napari.layers.labels.labels.Labels):
        if current_layer.show_selected_label:
            current_layer.show_selected_label = False
        else:
            current_layer.show_selected_label = True

@napari.Viewer.bind_key('e')
def edge_coloring(viewer):
    current_layer = viewer.layers.selection.active
    if isinstance(current_layer,napari.layers.labels.labels.Labels):
        current_layer.contour = 1

@napari.Viewer.bind_key('q')
def preserve_label(viewer):
    current_layer = viewer.layers.selection.active
    if isinstance(current_layer,napari.layers.labels.labels.Labels):
        current_layer.preserve_labels = not current_layer.preserve_labels

@napari.Viewer.bind_key('w')
def whole_coloring(viewer):
    current_layer = viewer.layers.selection.active
    if isinstance(current_layer,napari.layers.labels.labels.Labels):
        current_layer.contour = 0

@napari.Viewer.bind_key('b')
def large_brush(viewer):
    current_layer = viewer.layers.selection.active
    if isinstance(current_layer,napari.layers.labels.labels.Labels):
        current_layer.brush_size = 20
@napari.Viewer.bind_key('s')
def small_brush(viewer):
    current_layer = viewer.layers.selection.active
    if isinstance(current_layer,napari.layers.labels.labels.Labels):
        current_layer.brush_size = 5

@napari.Viewer.bind_key('g')
def find_labelID_in_stack(viewer):
    current_layer = viewer.layers.selection.active
    if isinstance(current_layer,napari.layers.labels.labels.Labels):
        currentID = current_layer.selected_label
        mask = current_layer.data == currentID

        # set the first axis to the axis where object is found at first
        where_in_first_axis = np.any(mask,axis=1)
        for i in range(where_in_first_axis.ndim - 1):
            where_in_first_axis = np.any(where_in_first_axis,axis=1)
        if where_in_first_axis.sum() > 0:
            viewer.dims.set_point(0,np.where(where_in_first_axis)[0][0])
        else:
            viewer.status='Current label not found!'

        # set the second axis (presumed Z) to the 'fattest slice'
        mask = np.moveaxis(mask,0,-1)
        where_in_second_axis = np.sum(mask,axis=1)
        for i in range(where_in_second_axis.ndim - 1):
            where_in_second_axis = np.sum(where_in_second_axis,axis=1)
        viewer.dims.set_point(1,where_in_second_axis.argmax())

@napari.Viewer.bind_key('/')
def cycle_active_axis(viewer):
    current_axis = viewer.dims.last_used
    viewer.dims.last_used = (current_axis + 1) % viewer.dims.ndisplay

dirnames = {'R1':'/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/',
           'R2':'/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'}

tracked_cyto_by_region = {name: io.imread(path.join(dirname,'Mastodon/tracked_cyto.tif')) for name,dirname in dirnames.items()}
tracked_nuc_by_region = {name: io.imread(path.join(dirname,'Mastodon/tracked_nuc.tif')) for name,dirname in dirnames.items()}
adjdict_by_region = {name: [np.load(path.join(dirname,f'Mastodon/basal_connectivity_3d/adjacenct_trackIDs_t{t}.npy'),allow_pickle=True).item() for t in range(15)] for name,dirname in dirnames.items()}

dataset_dir = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/Lineage models/Dataset pickles'
all_df = pd.read_pickle(path.join(dataset_dir,f'all_df.pkl'))
trackIDs = all_df.reset_index()['TrackID'].unique()
standard_size = (40,150,150)

@magicgui(track_to_show = {'choices':natsorted(trackIDs.tolist()) })
def load_track_mask(track_to_show=trackIDs.tolist()[0]):

    nuc_movie, cell_movie, micro_movie = get_nuc_and_cell_and_microenvironment_movie(track_to_show,all_df,
                                                    adjdict_by_region,
                                                    tracked_nuc_by_region,
                                                    tracked_cyto_by_region,
                                                    standard_size)

    viewer.add_labels(np.stack(nuc_movie),scale=scale,opacity=1,name='nucleus')
    viewer.add_labels(np.stack(cell_movie),scale=scale,opacity=0.6,name='cell')
    viewer.add_labels(np.stack(micro_movie),scale=scale,opacity=0.3,name='microenvironment')

dz = 1
dx = .25
scale = [dz,dx,dx]

viewer = napari.Viewer()

# viewer.add_tracks(tracks.values,scale = [1,dx,dx])
viewer.window.add_dock_widget(load_track_mask, name="Show microenvironment movie",area='left')
