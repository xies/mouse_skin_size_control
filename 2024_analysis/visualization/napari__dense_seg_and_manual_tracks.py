#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:58:49 2024

@author: xies
"""

from napari_animation import Animation
import napari
import skimage
from os import path
import numpy as np

viewer = napari.Viewer(ndisplay=3)
viewer.window.resize(1346,1046)

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

t = 0

cortex_data = skimage.io.imread(path.join(dirname,'Cropped_images/G.tif'))
nuc_data = skimage.io.imread(path.join(dirname,'Cropped_images/B.tif'))
# labels_data = skimage.io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}.tif'))
manual_tracking = skimage.io.imread(path.join(dirname,f'manual_basal_tracking/basal_tracks_cyto.tif'))

# Layers
nuc_layer = viewer.add_image(nuc_data,name = 'nuc',blending = 'additive', depiction = 'volume',
                             rendering='translucent',colormap='blue',scale=[1,.25,.25])
cortex_layer = viewer.add_image(cortex_data,name = 'cortex',blending = 'additive', depiction='volume',
                                rendering='translucent',scale=[1,.25,.25])
tracks_layer = viewer.add_labels(manual_tracking, name='tracks', blending='additive',scale=[1,.25,.25])

animation = Animation(viewer)

viewer.camera.angles = (-18.23797054423494, 41.97404742075617, 141.96173085742896)
viewer.camera.zoom *= 0.6

# Timestampper
def update_timestamp():
    """Update fps."""
    t = viewer.dims.current_step[0]
    viewer.text_overlay.color = [1,1,1]
    viewer.text_overlay.position = 'bottom_right'
    viewer.text_overlay.font_size=36
    viewer.text_overlay.text = f'Day {t/2}'

viewer.dims.current_step = (0,35,229,229)
viewer.dims.events.current_step.connect(update_timestamp)

# Hold on a blank movie
viewer.text_overlay.visible = False
nuc_layer.visible = False
cortex_layer.visible = False
tracks_layer.visible = False
animation.capture_keyframe(steps=15)

# Start movie
viewer.text_overlay.visible = True
nuc_layer.visible = True
cortex_layer.visible = True
tracks_layer.visible = True
animation.capture_keyframe(steps=30)

# Go through time positions 0->14
viewer.dims.current_step = (14,35,229,229)
animation.capture_keyframe(steps=120)

animation.capture_keyframe(steps=15)


animation.animate(path.join(dirname,'Examples for figures/Basal tracking 3D movie/dense_cyto_t0_to_basal_tracks.mp4'), canvas_only=True)




