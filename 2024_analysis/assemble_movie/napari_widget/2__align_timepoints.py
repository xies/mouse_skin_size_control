#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:22:11 2024

@author: xies
"""

from magicgui import magicgui
import napari
from typing import List, Tuple

from napari.layers import Image, Layer
from napari.utils.notifications import show_warning, show_info
from napari.utils import progress

# from os import path
import numpy as np
import math
from numpy.linalg import norm
from skimage import io,util,filters
from skimage.transform import EuclideanTransform, warp
from scipy import ndimage

from pystackreg import StackReg
from twophotonUtils import parse_unaligned_channels, find_most_likely_z_slice_using_CC, \
    z_align_ragged_timecourse, parse_aligned_timecourse_directory
from imageLoadingWidgets import LoadChannelForInspection

from pathlib import Path
from os import path
from glob import glob
from re import findall
import pickle as pkl

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
        B_ref = io.imread(filelist.loc[reference_timepoint,'B'])

        # Initiate transform matrices
        z_pos_in_original = {}
        XY_matrices = {}
        if path.exists(path.join(dirname,'alignment_information.pkl')):
            with open(path.join(dirname,'alignment_information.pkl'),'rb') as f:
                [z_pos_in_original,XY_matrices,Imax_ref] = pkl.load(f)

        # Select the reference z-slice: for mem-GFP, use a half-way point
        stds = np.array([X.std() for X in B_ref])
        z_ref = stds.argmax()
        # z_ref = 18
        ref_img = B_ref[z_ref,...]
        print(f'Reference z-slice: {z_ref}')
        output_imgs = [Image(ref_img)]

        ragged_B_stacks = []
        ragged_G_stacks = []
        ragged_R_stacks = []
        ragged_R_shg_stacks = []

        for t in progress(timepoints_to_register):

            # Check for overwriting
            print(reference_timepoint)
            if t == reference_timepoint:
                print(f'Skipping t = {t} because it is the reference')
                continue
            if path.exists(path.join(path.dirname(filelist.loc[t,'R_shg']),'R_shg_align.tif')) and not OVERWRITE:
                print(f'Skipping t = {t} because its R_shg_align.tif already exists')
                continue

            output_dir = path.dirname(filelist.loc[t,'R'])
            # Alignment code here
            print(f'\n ---- Working on t = {t} ----')
            #Load the target
            B_target = io.imread(filelist.loc[t,'B']).astype(float)

            # Find simlar in the next time point
            # 1. Use xcorr2 to find the z-slice on the target that has max CC with the reference
            z_target = find_most_likely_z_slice_using_CC(ref_img,B_target)
            print(f'Target z-slice automatically determined to be {z_target}')
            z_pos_in_original[t] = z_target

            target_img = B_target[z_target]
            output_imgs.append(Image(target_img,name=f'{t}_before'))

            # Save z-target

            # 2. Calculate XY transforms
            print('StackReg + transform')
            sr = StackReg(StackReg.RIGID_BODY)
            T = sr.register(target_img,ref_img) #Obtain the transformation matrices
            T = EuclideanTransform(T)

            # Load other channels + apply transformations
            B = io.imread(filelist.loc[t,'B']).astype(float)
            G = io.imread(filelist.loc[t,'G']).astype(float)
            R = io.imread(filelist.loc[t,'R']).astype(float)
            R_shg = io.imread(filelist.loc[t,'R_shg']).astype(float)

            B_transformed = np.zeros_like(B).astype(float)
            G_transformed = np.zeros_like(G).astype(float)
            R_transformed = np.zeros_like(R).astype(float)
            R_shg_transformed = np.zeros_like(R_shg).astype(float)
            for i, B_slice in enumerate(B):
                B_transformed[i,...] = warp(B_slice,T)
                G_transformed[i,...] = warp(G[i,...],T)
                R_transformed[i,...] = warp(R[i,...],T)
                R_shg_transformed[i,...] = warp(R_shg[i,...],T)

            # Save the 'ragged' zstacks
            ragged_B_stacks.append(B)
            ragged_G_stacks.append(G)
            ragged_R_stacks.append(R)
            ragged_R_shg_stacks.append(R_transformed)
            # z-pad
            # B_padded = z_translate_and_pad(B_ref,B_transformed,z_ref,z_target).astype(np.uint)
            # G_padded = z_translate_and_pad(B_ref,G_transformed,z_ref,z_target).astype(np.uint)
            # R_padded = z_translate_and_pad(B_ref,R_transformed,z_ref,z_target).astype(np.uint)
            # R_shg_padded = z_translate_and_pad(B_ref,R_shg_transformed,z_ref,z_target).astype(np.uint)



        B_aligned_ZXY = z_align_ragged_timecourse(ragged_B_stacks, np.array(z_pos_in_original.values()))
        G_aligned_ZXY = z_align_ragged_timecourse(ragged_G_stacks, np.array(z_pos_in_original.values()))
        R_aligned_ZXY = z_align_ragged_timecourse(ragged_R_stacks, np.array(z_pos_in_original.values()))
        R_shg_aligned_ZXY = z_align_ragged_timecourse(ragged_R_shg_stacks, np.array(z_pos_in_original.values()))

        for t in timepoints_to_register:

            output_dir = path.dirname(filelist.loc[t,'R'])
            # Save transformation matrix for display later
            # Save images directly
            print(f'Saving to {output_dir}')
            im = B_aligned_ZXY[t,...]
            io.imsave(path.join(output_dir,'B_align.tif'),util.img_as_uint(im/im.max()),check_contrast=False)
            im = G_aligned_ZXY[t,...]
            io.imsave(path.join(output_dir,'G_align.tif'),util.img_as_uint(im/im.max()),check_contrast=False)
            im = R_aligned_ZXY[t,...]
            io.imsave(path.join(output_dir,'R_align.tif'),util.img_as_uint(im/im.max()),check_contrast=False)
            im = R_shg_aligned_ZXY[t,...]
            io.imsave(path.join(output_dir,'R_shg_align.tif'),util.img_as_uint(im/im.max()),check_contrast=False)

            output_imgs.append(Image(im[z_target],name=f'{t}_after'))

        return output_imgs

    @widget.pattern_str.changed.connect
    @widget.dirname.changed.connect
    def update_timepoints_on_file_change(event=None):
        _update_timepoints_on_file_change(widget)

    return widget

def estimate_translation_and_rotation_from_anchors(A,B):
    '''
    See: https://lucidar.me/en/mathematics/calculating-the-transformation-between-two-set-of-points/
    @todo: extendable to 3D
    '''

    assert(len(A) == len(B)) # size must match
    assert(A.shape[1] == 2) # 2D anchors
    comA = A.mean(axis=0)
    comB = B.mean(axis=0)

    Ac = A - comA[None,:] # implicit repmat
    Bc = B - comB[None,:]

    M = np.stack([np.outer(Ac[i,:], Bc[i,:]) for i in range(len(A))]).sum(axis=0)
    U,S,V = np.linalg.svd(M)
    R = np.matmul(V,U.T)

    return comA,comB,R

# Mouse-selected
@magicgui(call_button='Transform image')
def transform_image(
    stack2transform: Image,
    reference_index: int=0,
    directory_to_save_transformations: Path=Path.home()
    ) -> Image:

    # Grab the image data
    image_data = stack2transform.data.astype(float)
    reference_image = image_data[reference_index,...]

    # Grab the reference point layers
    ref_point_name = stack2transform.name + '_ref_A'
    if ref_point_name in [l.name for l in viewer.layers]:
        anchors_A = viewer.layers[ref_point_name].data
    rotation_point_name = stack2transform.name + '_ref_B'
    if rotation_point_name in [l.name for l in viewer.layers]:
        anchors_B = viewer.layers[rotation_point_name].data

    # initialize the reference anchors for the right matrix algebra shape
    reference_cloud = np.stack([ anchors_A[reference_index,2:4], anchors_B[reference_index,2:4]])
    reference_cloud = np.squeeze(reference_cloud)
    # Always use AnchorA for z-anchor
    reference_z = anchors_A[reference_index,...][1].astype(int)
    output_stack = np.zeros_like(image_data)

    # Go through each time point
    transformations = dict()
    for t,anchor_A in enumerate(anchors_A):

        # Grab current image
        this_im = image_data[t,...]
        # IF this is is the refence point, put in original image and skip
        if t == reference_index:
            output_stack[t,...] = util.img_as_uint(this_im/this_im.max())
            transformations[t] = [reference_z,0,0,0]
            continue

        # First, z-translate (always use anchorA for now)
        moving_z = anchors_A[t,...][1].astype(int)
        array = z_translate_and_pad(reference_image,this_im,reference_z,moving_z)

        # Now, calculate translation and rotation together, disregarding z
        moving_cloud = np.stack([ anchors_A[t,2:4], anchors_B[t,2:4]])
        moving_cloud = np.squeeze(moving_cloud)

        com_ref,com_moving,Rm = estimate_translation_and_rotation_from_anchors(reference_cloud,moving_cloud)

        dy,dx = com_moving - com_ref
        print(f'dy={dy} and dx={dx}')
        theta = np.arctan2(Rm[0,1],Rm[0,0])

        Txy = EuclideanTransform(rotation=theta, translation= [dx,dy])
        for z,im in enumerate(array):
            array[z,...] = warp(im,Txy)

        # Txy = EuclideanTransform(rotation=theta)
        output_stack[t,...] = util.img_as_uint(array/array.max())
        transformations[t] = [moving_z,dx,dy,theta]

    #@todo: Save matrix?
    with open(path.join(directory_to_save_transformations,'manual.pkl'),'wb+') as f:
        pkl.dump(transformations,f)

    return( Image(output_stack,name='Manual') )

@magicgui(call_button='Refine alignment')
def refine_alignment(
        im2refine : Image,
        reference_index : int=0,
        z_to_use : int=10,
        directory_to_save_transformations : Path=Path.home()
        ) -> Image:
    im_stack = im2refine.data
    assert(im_stack.ndim == 4)

    TT,ZZ,YY,XX = im_stack.shape
    assert(z_to_use < ZZ)

    reference_stack = im_stack[reference_index,...]
    ref_img = reference_stack[z_to_use,...]
    array = np.zeros_like(im_stack)
    transformations = dict()
    for t in range(TT):
        this_im = im_stack[t,...]
        if t == reference_index:
            array[t,...] = im_stack[t,...]
            transformations[t] = [0,0,0]
            continue
        moving_img = this_im[z_to_use,...]
        sr = StackReg(StackReg.RIGID_BODY)
        T = sr.register(ref_img,moving_img) #Obtain the transformation matrices
        T = EuclideanTransform(T)

        for z,im in enumerate(this_im):
            array[t,z,...] = warp(im,T)
        transformations[t] = [T.translation[0],T.translation[1],T.rotation]

    with open(path.join(directory_to_save_transformations,'refinements.pkl'),'wb+') as f:
        pkl.dump(transformations,f)

    return(Image(array,name='Refined'))

@magicgui(call_button='Apply transformations to stack')
def apply_transformations_to_stack(
        im2transform : Image,
        directory2load: Path = Path.home(),
        apply_refinement: bool = True
    ) -> Image :

    im_stack = im2transform.data
    with open(path.join(directory2load,'manual.pkl'),'rb') as f:
        manual = pkl.load(f)

    if apply_refinement:
        with open(path.join(directory2load,'refinements.pkl'),'rb') as f:
            refinements = pkl.load(f)

    for t,(dz,dx,dy,theta) in manual.items():
        if dx == 0 and dy == 0 and theta == 0:
            reference_t = t
            reference_z = dz
    reference_image = im_stack[reference_t,...]

    array = np.zeros_like(im_stack)
    for t,this_im in enumerate(im_stack):
        moving_z,dx,dy,theta = manual[t]
        if t == reference_t:
            array[t,...] = this_im
            continue

        this_im = z_translate_and_pad(reference_image,this_im,reference_z,moving_z)
        T = EuclideanTransform(translation=[dx,dy],rotation=theta)

        for z,im in enumerate(this_im):
            array[t,z,...] = warp(im,T)

        if apply_refinement:
            dx,dy,theta = refinements[t]
            T = EuclideanTransform(translation=[dx,dy],rotation=theta)

            for z,im in enumerate(this_im):
                array[t,z,...] = warp(im,T)

    return(Image(array,name=f'{im2transform.name}_aligned'))

viewer = napari.Viewer()

viewer.window.add_dock_widget(auto_align_timecourse(),area='left')
viewer.window.add_dock_widget(LoadChannelForInspection(viewer),area='right')
viewer.window.add_dock_widget(transform_image,area='right')
viewer.window.add_dock_widget(refine_alignment,area='right')
viewer.window.add_dock_widget(apply_transformations_to_stack,area='right')

if __name__ == '__main__':
    napari.run()
