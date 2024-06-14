#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:22:11 2024

@author: xies
"""

from magicgui import magicgui
import napari
from typing import List

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
    pattern_str = widget.pattern_str.value
    choices = None
    filelist = parse_unaligned_channels(dirname, folder_str=pattern_str)

    if len(filelist) > 0:
        choices = filelist.index
    else:
        show_warning(f'No timepoints found in {dirname}/{pattern_str}')
    if choices is not None:
        widget.timepoints_to_register.choices = choices
        widget.reference_timepoint.choices = choices

def auto_align_timecourse():
    '''
    Gets dirname from picker and populates the available timepoints to align.
    See _update_timepoints_on_file_change for the populator.
    Auto aligns time course using selected reference time point
    '''
    @magicgui(call_button='Align time course',
              reference_timepoint={'widget_type':'ComboBox',
                                      'choices':DEFAULT_CHOICES,
                                      'label':'Reference image',
                                      'allow_multiple':True},
              timepoints_to_register={'widget_type':'Select',
                                      'choices':DEFAULT_CHOICES,
                                      'label':'Time pionts to align',
                                      'allow_multiple':True},
              dirname={'label':'Image region to load:','mode':'d'})
    def widget(
        dirname=Path.home(),
        pattern_str='*.*/',
        reference_timepoint=0,
        timepoints_to_register=[0],
        OVERWRITE: bool=False,
        ) -> List[napari.layers.Layer]:

        filelist = parse_unaligned_channels(dirname,folder_str=pattern_str)

        # Load reference image
        print(timepoints_to_register)
        R_shg_ref = io.imread(filelist.loc[reference_timepoint,'R_shg'])

        # Initiate transform matrices
        z_pos_in_original = {}
        XY_matrices = {}
        if path.exists(path.join(dirname,'alignment_information.pkl')):
            with open(path.join(dirname,'alignment_information.pkl'),'rb') as f:
                [z_pos_in_original,XY_matrices,Imax_ref] = pkl.load(f)

        # Select the reference z-slice: for mem-GFP, use a half-way point
        z_ref = R_shg_ref.shape[0] // 2
        z_ref = 18
        ref_img = R_shg_ref[z_ref,...]
        print(f'Reference z-slice: {z_ref}')
        output_imgs = [Image(ref_img)]

        for t in progress(timepoints_to_register):

            # Check for overwriting
            print(reference_timepoint)
            if t == reference_timepoint:
                print(f'Skipping t = {t} because it is the reference')
                continue
            if path.exists(path.join(path.dirname(filelist.loc[t,'R_shg']),'R_shg_align.tif'))  and not OVERWRITE:
                print(f'Skipping t = {t} because its R_shg_align.tif already exists')
                continue

            output_dir = path.dirname(filelist.loc[t,'R'])
            # Alignment code here
            print(f'\n ---- Working on t = {t} ----')
            #Load the target
            R_shg_target = io.imread(filelist.loc[t,'R_shg']).astype(float)

            # Find simlar in the next time point
            # 1. Use xcorr2 to find the z-slice on the target that has max CC with the reference
            z_target = find_most_likely_z_slice_using_CC(ref_img,R_shg_target)
            print(f'Target z-slice automatically determined to be {z_target}')
            z_pos_in_original[t] = z_target

            target_img = R_shg_target[z_target]
            output_imgs.append(Image(target_img,name=f'{t}_before'))

            # 2. Calculate XY transforms
            print('StackReg + transform')
            sr = StackReg(StackReg.RIGID_BODY)
            T = sr.register(target_img/target_img.max(),ref_img) #Obtain the transformation matrices
            T = EuclideanTransform(T)

            # Load other channels + apply transformations
            B = io.imread(filelist.loc[t,'B']).astype(float)
            G = io.imread(filelist.loc[t,'G']).astype(float)
            R = io.imread(filelist.loc[t,'R']).astype(float)

            B_transformed = np.zeros_like(B).astype(float)
            G_transformed = np.zeros_like(G).astype(float)
            R_transformed = np.zeros_like(R).astype(float)
            R_shg_transformed = np.zeros_like(R_shg_target).astype(float)
            for i, B_slice in enumerate(B):
                B_transformed[i,...] = warp(B_slice,T)
                G_transformed[i,...] = warp(G[i,...],T)
                R_transformed[i,...] = warp(R[i,...],T)
                R_shg_transformed[i,...] = warp(R_shg_target[i,...],T)

            # z-pad
            B_padded = z_translate_and_pad(R_shg_ref,B_transformed,z_ref,z_target).astype(np.uint)
            G_padded = z_translate_and_pad(R_shg_ref,G_transformed,z_ref,z_target).astype(np.uint)
            R_padded = z_translate_and_pad(R_shg_ref,R_transformed,z_ref,z_target).astype(np.uint)
            R_shg_padded = z_translate_and_pad(R_shg_ref,R_shg_transformed,z_ref,z_target).astype(np.uint)

            # Save transformation matrix for display later
            # Save images directly

            print(f'Saving to {output_dir}')
            io.imsave(path.join(output_dir,'B_align.tif'),util.img_as_uint(B_padded/B_padded.max()),check_contrast=False)
            io.imsave(path.join(output_dir,'G_align.tif'),util.img_as_uint(G_padded/G_padded.max()),check_contrast=False)
            io.imsave(path.join(output_dir,'R_align.tif'),util.img_as_uint(R_padded/R_padded.max()),check_contrast=False)
            io.imsave(path.join(output_dir,'R_shg_align.tif'),util.img_as_uint(R_shg_padded/R_shg_padded.max()),check_contrast=False)

            output_imgs.append(Image(R_shg_target[z_target],name=f'{t}_after'))

        return output_imgs

    @widget.pattern_str.changed.connect
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

viewer.window.add_dock_widget(auto_align_timecourse(),area='left')
viewer.window.add_dock_widget(LoadAlignedChannelForInspection(viewer),area='right')

if __name__ == '__main__':
    napari.run()
