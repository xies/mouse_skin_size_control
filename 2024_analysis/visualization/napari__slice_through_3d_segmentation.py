#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:14:40 2024

@author: xies

based on: https://github.com/napari/napari-animation/blob/main/examples/layer_planes.py

"""

from napari_animation import Animation
import napari
import skimage
from os import path

viewer = napari.Viewer(ndisplay=3)
viewer.window.resize(1346,1046)

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

t = 0

cortex_data = skimage.io.imread(path.join(dirname,'Cropped_images/G.tif'))
nuc_data = skimage.io.imread(path.join(dirname,'Cropped_images/B.tif'))
labels_data = skimage.io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}.tif'))
# labels_data = skimage.io.imread(path.join(dirname,f'3d_cyto_seg/nucID_label_transfered/t{t}.tif'))


nuc_layer = viewer.add_image(nuc_data,name = 'nuc',blending = 'translucent', depiction = 'plane', colormap='blue',scale=[1,.25,.25])
cortex_layer = viewer.add_image(cortex_data,name = 'cortex',blending = 'additive', depiction='plane',scale=[1,.25,.25])
labels_layer = viewer.add_labels(labels_data,name = 'labels',blending = 'translucent',scale=[1,.25,.25])

animation = Animation(viewer)

viewer.camera.angles = (-18.23797054423494, 41.97404742075617, 141.96173085742896)
viewer.camera.zoom *= 0.6


# Tie the truncation point to the slice position of nuc_layer
def replace_labels_data():
    z_cutoff = int(nuc_layer.plane.position[0])
    new_labels_data = labels_data.copy()
    new_labels_data[z_cutoff : ,...] = 0
    labels_layer.data = new_labels_data


# Start at top
# nuc_layer.plane.position = (0, 0, 0)
# cortex_layer.plane.position = (0, 0, 0)

# animation.capture_keyframe(steps=30)

# Now clip the segmentation until the current plane and go down again, 2x slower
# nuc_layer.plane.events.position.connect(replace_labels_data)
# labels_layer.visible = True
# labels_layer.experimental_clipping_planes = [{
#     "position": (0, 0, 0),
#     "normal": (-1, 0, 0),  # point up in z (i.e: show stuff above plane)
# }]

# nuc_layer.plane.position = (71, 0, 0)
# cortex_layer.plane.position = (71, 0, 0)
# # access first plane, since it's a list
# labels_layer.experimental_clipping_planes[0].position = (71, 0, 0)
# animation.capture_keyframe(steps=60)

# How show entire image as a block with seg on top and hold

# nuc_layer.plane.position = (71, 0, 0)
nuc_layer.depiction='volume'
cortex_layer.depiction='volume'
nuc_layer.rendering = 'translucent'
cortex_layer.rendering = 'translucent'
labels_layer.visible=False

viewer.camera.angles = (-18.23797054423494, 41.97404742075617, 141.96173085742896)
animation.capture_keyframe(steps=60)

viewer.camera.angles = (15, -28, 141.96173085742896)
animation.capture_keyframe(steps=60)

viewer.camera.angles = (-18.23797054423494, 41.97404742075617, 141.96173085742896)
animation.capture_keyframe(steps=60)

# Seg pops out on top and hold
labels_layer.visible=True
labels_layer.blending = 'additive'

animation.capture_keyframe(steps=10)

# Rock back and forth

viewer.camera.angles = (15, -28, 141.96173085742896)
animation.capture_keyframe(steps=60)

viewer.camera.angles = (-18.23797054423494, 41.97404742075617, 141.96173085742896)
animation.capture_keyframe(steps=60)

animation.capture_keyframe(steps=10)

# animation.animate(path.join(dirname,'Examples for figures/Segmentation_slice_through/t0_nuc.mp4'), canvas_only=True)
animation.animate('/Users/xies/Desktop/t0.mp4',canvas_only=True)



