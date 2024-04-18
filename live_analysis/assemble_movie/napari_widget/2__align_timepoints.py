#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:22:11 2024

@author: xies
"""

from magicgui import magicgui
import napari

import napari
from napari.layers import Image
from napari.utils.notifications import show_warning, show_info
from napari.utils import progress

# from os import path
import numpy as np
from skimage import io,util,filters
from skimage.transform import EuclideanTransform, warp
from scipy import ndimage

from pystackreg import StackReg
from twophotonUtils import parse_unaligned_channels, find_most_likely_z_slice_using_CC, \
    z_translate_and_pad, parse_aligned_timecourse_directory
from imageLoadingWidgets import LoadAlignedChannelForInspection, LoadDTimepointForInspection

from pathlib import Path
from os import path
from glob import glob
from re import findall

DEFAULT_CHOICES = [0]

@magicgui(call_button='Print size')
def print_image_size(image : Image):
    show_info(f'Size of {image.name} = {image.data.shape}')

def _update_timepoints_on_file_change(widget):
    """Called whenever the file picker is changed. Will look into that directory,
    and return the available timepoints as defined by parse_unregistered_channels
    """
    
    dirname = widget.dirname.value
    choices = None
    filelist = parse_unaligned_channels(dirname)
    
    if len(filelist) > 0:
        choices = filelist.index
    else:
        show_warning(f'Directory {dirname} is not a region directory.')
    if choices is not None:
        widget.timepoints_to_align.choices = choices
    
def auto_align_timecourse():
    '''
    Gets dirname from picker and populates the available timepoints to align.
    See _update_timepoints_on_file_change for the populator.
    Auto aligns time course using selected reference time point
    '''
    @magicgui(call_button='Align time course',
              timepoints_to_register={'widget_type':'Select',
                                      'choices':DEFAULT_CHOICES,
                                      'label':'Time points to align'
                                      'allow_multiple':True},
              reference_timepoint={'widget_type':'Select',
                                      'choices':DEFAULT_CHOICES,
                                      'label':'Time points to align'
                                      'allow_multiple':True},
              dirname={'label':'Image region to load:','mode':'d'})
    def widget(
        dirname=Path.home(),
        timepoints_to_register=(0),
        reference_timepoint = 
        OVERWRITE: bool=False,
        ):
    
        filelist = parse_unaligned_channels(dirname)
        
        for t in progress(timepoints_to_register):
        
            # Check for overwriting
            if path.exists(path.join(path.dirname(filelist.loc[t,'R_shg']),'R_shg_align.tif'))  and not OVERWRITE:
                print(f'Skipping t = {t} because its R_shg_align.tif already exists')
                continue
            
            # Alignment code here
            # Save transformation matrix for display later
            # Save images directly
            
    @widget.dirname.changed.connect
    def update_timepoints_on_file_change(event=None):
        _update_timepoints_on_file_change(widget)
    return widget


@magicgui(call_button='Transform image')
def transform_image(
    reference_image: Image,
    image2transform: Image,
    second_channel: Image,
    reference_z: int=0,
    reference_y: int=0,
    reference_x: int=0,
    moving_z: int=0,
    moving_y: int=0,
    moving_x: int=0,
    rotate_theta:float=0.0,
    Transform_second_channel:bool=False,
    ) -> List[napari.layers.Layer]:
    
    '''
    Perform 3D rigid-body transformations given an input image and manually set transformation parameters
    translate z/y/x: pixel-wise translations
    rotate_theta: rotation angle in degrees

    Optionally, select a second channel to also transform with the same transformation matrix

    Output will be the transformed image(s), with _transformed appended to name(s)
    '''
    
    # Grab the image data + convert deg->radians
    image_data = image2transform.data.astype(float)
    second_image_data = second_channel.data.astype(float)
    rotate_theta = np.deg2rad(rotate_theta)

    # xy transformations (do slice by slice)
    Txy = EuclideanTransform(translation=[moving_x-reference_x,moving_y-reference_y], rotation=rotate_theta)
    # # Apply to first image
    array = np.zeros_like(image_data)
    for z,im in enumerate(image_data):
        array[z,...] = warp(im, Txy)
    array = z_translate_and_pad(reference_image.data,array,reference_z,moving_z)
    
    transformed_image = Image(array, name=image2transform.name+'_transformed', blending='additive', colormap=image2transform.colormap)
    output_list = [transformed_image]
    
    if Transform_second_channel:
        array = np.zeros_like(second_image_data)
        for z,im in enumerate(second_image_data):
            array[z,...] = warp(im, Txy)
        array = z_translate_and_pad(reference_image.data,array,reference_z,moving_z)
        transformed_second_channel = Image(array, name=second_channel.name+'_transformed', blending='additive', colormap=second_channel.colormap)
        output_list.append(transformed_second_channel)
        
    return output_list

viewer = napari.Viewer()

viewer.window.add_dock_widget(print_image_size, area='left')

# viewer.window.add_dock_widget(load_ome_tiffs_and_save_as_tiff,area='left')
# viewer.window.add_dock_widget(auto_register_b_and_rshg(),area='left')
viewer.window.add_dock_widget(LoadAlignedChannelForInspection(viewer),area='right')
viewer.window.add_dock_widget(LoadDTimepointForInspection(viewer),area='right')



if __name__ == '__main__':
    napari.run()
