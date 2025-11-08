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


# Load the images
dirnames = ['/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/W-R1/',
    '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/W-R2/']


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

@magicgui(dirname ={'choices':dirnames,
        'allow_multiple':False})
def load_dataset(dirname = (dirnames[0])):

    names2remove = [l.name for l in viewer.layers]
    for l in names2remove:
        self._viewer.layers.remove(l)

    # df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints.csv'))
    # measurement_list = df.columns[(~df.columns.str.startswith('cyto_')) & (~df.columns.str.startswith('nuc_'))].tolist()
    # Load the manual tracks
    viewer.
    all_df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints.csv'),index_col=0).reset_index()
    all_df = all_df[all_df['Cell type'] == 'Basal']
    # Tracks axes are: ID,T,(Z),Y,X
    tracks = all_df[all_df['Cell type'] == 'Suprabasal'][['TrackID','Frame','Z','Y-pixels','X-pixels']]

    with open(path.join(dirname,'Mastodon/dense_tracks.pkl'),'rb') as file:
        tracks = pkl.load(file)
    tracks = pd.concat(tracks,ignore_index=True).sort_values(['LineageID','Frame'])
    # Sanitize dtype
    tracks = tracks[['LineageID','Frame','Z','Y','X']].astype(float)
    # The default output is in microns -> convert
    tracks['Y'] = tracks['Y'] / dx
    tracks['X'] = tracks['X'] / dx

    R = io.imread(path.join(dirname,'Cropped_images/R.tif'))
    B = io.imread(path.join(dirname,'Cropped_images/B.tif'))
    G = io.imread(path.join(dirname,'Cropped_images/G.tif'))
    segmentation = io.imread(path.join(dirname,'Mastodon/tracked_cyto.tif'))
    nuc_segmentation = io.imread(path.join(dirname,'Mastodon/tracked_nuc.tif'))
    connectivity = io.imread(path.join(dirname,'Mastodon/basal_connectivity_3d/basal_connectivity_3d.tif'))

    filelist = natsorted(glob(path.join(dirname,'Image flattening/height_image/t*.tif')))
    basement_mem = np.stack([io.imread(f) for f in filelist])
    viewer.add_image(R,scale = scale, blending='additive', colormap='red',rendering='attenuated_mip')
    viewer.add_image(B,scale = scale, blending='additive', colormap='blue',rendering='attenuated_mip')
    viewer.add_image(G,scale = scale, blending='additive', colormap='gray',visible=True,rendering='attenuated_mip')
    viewer.add_image(basement_mem,scale=[dz,dx,dx],blending='additive',colormap='gray',visible=True)
    viewer.add_labels(connectivity,scale = scale)
    viewer.add_labels(segmentation,scale = scale)
    viewer.add_labels(nuc_segmentation,scale = scale)




dz = 1
dx = .25
scale = [dz,dx,dx]

viewer = napari.Viewer()

# viewer.add_tracks(tracks.values,scale = [1,dx,dx])

viewer.window.add_dock_widget(load_dataset, name="Load dataset")
