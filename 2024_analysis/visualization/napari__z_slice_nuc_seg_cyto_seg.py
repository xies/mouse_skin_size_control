#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:51:15 2024

@author: xies
"""

from napari_animation import Animation
import napari
import skimage
from os import path

viewer = napari.Viewer(ndisplay=2)
viewer.window.resize(1346,1046)

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

t = 1

cortex_data = skimage.io.imread(path.join(dirname,'Cropped_images/G.tif'))[t,...]
nuc_data = skimage.io.imread(path.join(dirname,'Cropped_images/B.tif'))[t,...]
nuc_labels_data = skimage.io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}.tif'))
cyto_labels_data = skimage.io.imread(path.join(dirname,f'3d_cyto_seg/nucID_label_transfered/t{t}.tif'))


nuc_layer = viewer.add_image(nuc_data,name = 'nuc',blending = 'translucent', depiction = 'plane', colormap='blue',scale=[1,.25,.25])
cortex_layer = viewer.add_image(cortex_data,name = 'cortex',blending = 'additive', depiction='plane',scale=[1,.25,.25])
labels_layer = viewer.add_labels(nuc_labels_data,name = 'nuc_labels',blending = 'translucent_no_depth',scale=[1,.25,.25])
labels_layer.contour = 1
cyto_labels_layer = viewer.add_labels(cyto_labels_data,name = 'cyto_labels',blending = 'translucent_no_depth',scale=[1,.25,.25])
cyto_labels_layer.contour = 1

viewer.scale_bar.visible = True
viewer.scale_bar.unit = 'um'

animation = Animation(viewer)


# Go down the stack and show the segmentation as countours
viewer.dims.current_step = (0,229,229)
animation.capture_keyframe(steps=60)

viewer.dims.current_step = (71,229,229)
animation.capture_keyframe(steps=60)


animation.animate(path.join(dirname,f'Examples for figures/Dense seg movies/t{t}.mp4'), canvas_only=True)



