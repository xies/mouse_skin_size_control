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

region = 'R2'

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

    # # Tracks axes are: ID,T,(Z),Y,X
    # tracks = all_df[all_df['Cell type'] == 'Suprabasal'][['TrackID','Frame','Z','Y-pixels','X-pixels']]
    # with open(path.join(dirname,'Mastodon/dense_tracks.pkl'),'rb') as file:
    #     tracks = pkl.load(file)
    # tracks = pd.concat(tracks,ignore_index=True).sort_values(['LineageID','Frame'])
    # # Sanitize dtype
    # tracks = tracks[['LineageID','Frame','Z','Y','X']].astype(float)
    # # The default output is in microns -> convert
    # tracks['Y'] = tracks['Y'] / dx
    # tracks['X'] = tracks['X'] / dx

    R = io.imread(path.join(dirname,'Cropped_images/R.tif'))
    B = io.imread(path.join(dirname,'Cropped_images/B.tif'))
    G = io.imread(path.join(dirname,'Cropped_images/G.tif'))

    segmentation = io.imread(path.join(dirname,'Mastodon/tracked_cyto.tif'))
    nuc_segmentation = io.imread(path.join(dirname,'Mastodon/tracked_nuc.tif'))
    lin_cyto = io.imread(path.join(dirname,'Mastodon/lineageID_cyto.tif'))
    lin_nuc = io.imread(path.join(dirname,'Mastodon/lineageID_nuc.tif'))
    connectivity = io.imread(path.join(dirname,'Mastodon/basal_connectivity_3d/basal_connectivity_3d.tif'))

    filelist = natsorted(glob(path.join(dirname,'Image flattening/height_image/t*.tif')))
    basement_mem = np.stack([io.imread(f) for f in filelist])
    # macrophages
    macrophages = []
    for t in range(15):
        df_ = pd.read_csv(path.join(dirname,f'3d_cyto_seg/macrophages/t{t}.csv'))
        df_ = df_.rename(columns={'axis-0':'axis-1','axis-1':'axis-2','axis-2':'axis-3'})
        df_['axis-0'] = t
        macrophages.append(df_)
    macrophages = pd.concat(macrophages,ignore_index=True).drop(columns='index')
    macrophages = macrophages[['axis-0','axis-1','axis-2','axis-3']]
    macrophages['axis-2'] *= 0.25
    macrophages['axis-3'] *= 0.25
    print(macrophages)

    # Add surface
    data = np.load(path.join(dirname,f'Image flattening/trimesh/bg_surface_timeseries.npz'))
    vertices = data['vertices']
    faces = data['faces']
    values = data['values']

    viewer.add_image(R,scale = scale, blending='additive', colormap='red',rendering='attenuated_mip',visible=False)
    viewer.add_image(B,scale = scale, blending='additive', colormap='blue',rendering='attenuated_mip')
    viewer.add_image(G,scale = scale, blending='additive', colormap='gray',visible=True,rendering='attenuated_mip',attenuation=1)
    viewer.add_surface((vertices,faces,values),scale=[1,1,1],colormap='orange',opacity=0.6)
    viewer.add_image(basement_mem,scale=[dz,dx,dx],blending='additive',colormap='gray',visible=False)
    viewer.add_labels(connectivity,scale = scale,visible=False,blending='additive')
    viewer.add_labels(nuc_segmentation,scale = scale,name='nuclei',opacity=1,blending='additive')
    viewer.add_labels(segmentation,scale = scale,name='cytoplasms',blending='additive')
    viewer.add_labels(lin_nuc,scale = scale,name='lineage_nuc',opacity=1,blending='additive')
    viewer.add_labels(lin_cyto,scale = scale,name='lineage_cyto',blending='additive')

    viewer.add_points(macrophages,name='macrophages',face_color='red')


dirname = dirnames[0]
all_df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints_pca.csv'),index_col=0).reset_index()

lineageIDs = all_df['LineageID']

# @magicgui(lineage_to_show = {'choices':sorted(lineageIDs.tolist()) })
# def highlight_lineage(lineage_to_show=lineageIDs.tolist()[0]):
#
#     lineage = all_df[all_df['LineageID'] == lineage_to_show]
#
#     # Create masks
#     cyto = viewer.layers['cytoplasms'].data
#     nuc = viewer.layers['nuclei'].data
#     nuc_highlights = np.zeros_like(cyto)
#     cyto_highlights = np.zeros_like(cyto)
#
#     for _,row in lineage.iterrows():
#         t = row['Frame']
#         mask = (nuc[t,...] == row['TrackID'])
#         nuc_highlights[t,mask] = row['TrackID']
#         mask = (cyto[t,...] == row['TrackID'])
#         cyto_highlights[t,mask] = row['TrackID']
#
#     if 'highlighted_lineage_nuc' in viewer.layers:
#         viewer.layers.remove('highlighted_lineage_nuc')
#     if 'highlighted_lineage_cyto' in viewer.layers:
#         viewer.layers.remove('highlighted_lineage_cyto')
#     viewer.layers['cytoplasms'].visible = False
#     viewer.layers['nuclei'].visible = False
#
#     viewer.add_labels(cyto_highlights,name='highlighted_lineage_cyto',scale=scale,opacity=0.7,blending='additive')
#     viewer.add_labels(nuc_highlights,name='highlighted_lineage_nuc',scale=scale,opacity=1,blending='additive')



dz = 1
dx = .25
scale = [dz,dx,dx]

viewer = napari.Viewer()
lineageIDs = all_df['LineageID']

# viewer.add_tracks(tracks.values,scale = [1,dx,dx])
viewer.window.add_dock_widget(load_dataset, name="Load dataset",area='left')
# viewer.window.add_dock_widget(highlight_lineage, name="Show lineage",area='left')
