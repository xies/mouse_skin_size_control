#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:42:36 2024

@author: xies
"""

from pathlib import Path
from typing import List

from magicgui import magicgui
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
from twophotonUtils import parse_unregistered_channels, find_most_likely_z_slice_using_CC, z_translate_and_pad
from imageLoadingWidgets import LoadTimepointForInspection

from os import path
from glob import glob
from re import findall

# Default choices for the timepoints to insepct
global DEFAULT_CHOICES
DEFAULT_CHOICES = [None]

#%% Reading the first ome-tiff file using imread reads entire stack

# Extract the first ome.tiff file from every subfolder, load, then separate the two channels
def sort_by_slice(filename):
    z = findall('_(\d+).ome.tif',filename)[0]
    return int(z)



'''
Step one: Load all the OME-TIFFs and re-save as multipage TIFFs for each time point in region
Will perform a StackReg on each stack just in case there is small movement.
'''

@magicgui(call_button='Resave as TIFF',
          dirname={'label':'Image region to load:','mode':'d'})
def load_ome_tiffs_and_save_as_tiff(
    dirname=Path('/some/path.ext'),
    pattern_str: str='*/ZSeries*/',
    OVERWRITE: bool=False,
    register_G: bool=True,
    register_R: bool=True,
    ):

    subfolders = glob(path.join(dirname,pattern_str))

    header_ome_h2b = []
    header_ome_fucci = []
    for d in subfolders:
        ome_tifs = glob(path.join(d,'*.ome.tif'))
        ome_tifs = sorted(ome_tifs) # Sort by channel #
        ome_tifs = sorted(ome_tifs, key = sort_by_slice) # Sort by slice #
        if len(ome_tifs) < 40:
            print(f'Skipping {d}')
        else:
            if len(findall('1020nm',path.split(path.split(d)[0])[1])) == 0:
                header_ome_h2b.append(ome_tifs[0])
            else:
                header_ome_fucci.append(ome_tifs[0])

    # Register the B/G channels (using B as reference)
    for header_ome in progress(header_ome_h2b):

        d = path.split(path.dirname(header_ome))[0]
        # Make sure we haven't already processed this stack
        if path.exists(path.join(d,'G_reg.tif')) and not OVERWRITE:
            print(f'Skipping {d}')
            continue

        # Load ome-tif
        print(f'Loading {d}')
        stack = io.imread(header_ome,is_ome=True)
        if stack.ndim > 3:
            G = stack[0,...]
            B = stack[1,...]
        else:
            B = stack

        if register_G:
            # Use StackReg
            print(f'Registering {d}')
            sr = StackReg(StackReg.TRANSLATION) # There should only be slight sliding motion within a single stack
            T = sr.register_stack(B,reference='previous',n_frames=20,axis=0) #Obtain the transformation matrices
            B_reg = sr.transform_stack(B,tmats=T) # Apply to both channels
            G_reg = sr.transform_stack(G,tmats=T)
        else:
            B_reg = B
            G_reg = G

        output_path = path.join( d,'B_reg.tif')
        io.imsave(output_path,util.img_as_uint(B_reg/B_reg.max()),check_contrast=False)
        output_path = path.join( d,'G_reg.tif')
        io.imsave(output_path,util.img_as_uint(G_reg/G_reg.max()),check_contrast=False)

        print(f'Saved with {output_path}')

    # Register the R/R_shg channels (using R as reference)
    for header_ome in progress(header_ome_fucci):

        d = path.split(path.dirname(header_ome))[0]
        # Make sure we haven't already processed this stack
        if path.exists(path.join(d,'R_reg.tif')) and not OVERWRITE:
            print(f'Skipping {d}')
            continue

        # Load ome-tif
        print(f'Loading {d}')
        stack = io.imread(header_ome)
        R = stack[0,...]
        R_shg = stack[1,...]

        # Use StackReg
        if register_R:
            print(f'Registering {d}')
            sr = StackReg(StackReg.TRANSLATION) # There should only be slight sliding motion within a single stack
            T = sr.register_stack(R,reference='previous',axis=0) #Obtain the transformation matrices
            R_reg = sr.transform_stack(R,tmats=T) # Apply to both channels
            R_shg_reg = sr.transform_stack(R_shg,tmats=T) # Apply to both channels
        else:
            R_reg = R
            R_shg_reg = R_shg

        output_path = path.join( d,'R_reg.tif')
        io.imsave(output_path,util.img_as_uint(R_reg/R_reg.max()),check_contrast=False)
        output_path = path.join( d,'R_shg_reg.tif')
        io.imsave(output_path,util.img_as_uint(R_shg_reg/R_shg_reg.max()),check_contrast=False)

        print(f'Saved with {output_path}')


'''
Step two: automatically register the two channels for each timepoint
'''
def _update_timepoints_on_file_change(widget):
    """Called whenever the file picker is changed. Will look into that directory,
    and return the available timepoints as defined by parse_unregistered_channels
    """

    dirname = widget.dirname.value
    pattern_str = widget.pattern_str.value
    choices = None
    filelist = parse_unregistered_channels(dirname,pattern_str)

    if len(filelist) > 0:
        choices = filelist.index
    else:
        show_warning(f'Directory {dirname} is not a region directory.')
    if choices is not None:
        widget.timepoints_to_register.choices = choices

def auto_register_b_and_rshg():
    '''
    Gets dirname from picker and populates the available timepoints to register.
    See _update_timepoints_on_file_change for the populator
    '''
    @magicgui(call_button='Register channels (R_shg and B)',
              timepoints_to_register={'widget_type':'Select',
                                      'choices':DEFAULT_CHOICES,
                                      'label':'Time points to register'},
              dirname={'label':'Image region to load:','mode':'d'})
    def widget(
        dirname=Path.home(),
        pattern_str: str='*. Day*/',
        timepoints_to_register=None,
        OVERWRITE: bool=False,
        ):

        filelist = parse_unregistered_channels(dirname, folder_str=pattern_str)

        for t in progress(timepoints_to_register):

            # Check for overwriting
            output_dir = path.split(path.dirname(filelist.loc[t,'R']))[0]
            if path.exists(path.join(path.dirname(filelist.loc[t,'R']),'R_reg_reg.tif'))  and not OVERWRITE:
            # and path.exists(path.join(path.dirname(B_tifs[t]),'B_reg_reg.tif'))  and not OVERWRITE:
                print(f'Skipping t = {t} because its R_reg_reg.tif already exists')
                continue

            print(f'\n--- Started t = {t} ---')
            G = io.imread(filelist.loc[t,'G'])
            R_shg = io.imread(filelist.loc[t,'R_shg'])
            # G = io.imread(filelist.loc[t,'G'])
            R = io.imread(filelist.loc[t,'R'])
            print('Done reading images')

            # Find the slice with maximum mean value in R_shg channel
            z_ref = R_shg.mean(axis=2).mean(axis=1).argmax()
            z_ref = R_shg.shape[0] // 4
            print(f't = {t}: R_shg max std at {z_ref}')
            R_ref = R_shg[z_ref,...]
            R_ref = filters.gaussian(R_ref,sigma=0.5)
            z_moving = find_most_likely_z_slice_using_CC(R_ref,G)
            print(f'Cross correlation done and target Z-slice set at: {z_ref}')
            target = filters.gaussian(R_shg[z_ref,...],sigma=0.5)

            #NB: Here, move the R channel wrt the B channel
            print('StackReg + transform')
            sr = StackReg(StackReg.RIGID_BODY)
            T = sr.register(target/target.max(),R_ref) #Obtain the transformation matrices
            T = EuclideanTransform(T)
            R_transformed = np.zeros_like(R).astype(float)
            R_shg_transformed = np.zeros_like(R).astype(float)
            for i, R_slice in enumerate(R):
                R_transformed[i,...] = warp(R_slice,T)
                R_shg_transformed[i,...] = warp(R_shg[i,...],T)

            # z-pad
            print(R_transformed.shape)
            R_padded = z_translate_and_pad(G,R_transformed,z_ref,z_moving).astype(np.uint16)
            print(R_padded.shape)
            R_shg_padded = z_translate_and_pad(G,R_shg_transformed,z_ref,z_moving).astype(np.uint16)

            output_dir = path.dirname(filelist.loc[t,'G'])

            print('Saving')
            io.imsave(path.join(output_dir,'R_reg_reg.tif'),util.img_as_uint(R_padded/R_padded.max()),check_contrast=False)
            io.imsave(path.join(output_dir,'R_shg_reg_reg.tif'),util.img_as_uint(R_shg_padded/R_shg_padded.max()),check_contrast=False)

    @widget.pattern_str.changed.connect
    @widget.dirname.changed.connect
    def update_timepoints_on_file_change(event=None):
        _update_timepoints_on_file_change(widget)

    return widget

def swap_axes_order(im,current_order, new_order):
    swap_idx = {}
    swap_idx['X'] = current_order.index('X')
    swap_idx['Y'] = current_order.index('Y')
    swap_idx['Z'] = current_order.index('Z')
    swap_idx['C'] = current_order.index('C')

    im = np.transpose(im,[swap_idx[ new_order[0] ]
                        ,swap_idx[ new_order[1] ]
                        ,swap_idx[ new_order[2] ]
                        ,swap_idx[ new_order[3] ]])
    return im

def swap_axes_order_ref_point(ref_pt, current_order, new_order):
    swap_idx = {}
    swap_idx['X'] = current_order.index('X')
    swap_idx['Y'] = current_order.index('Y')
    swap_idx['Z'] = current_order.index('Z')
    swap_idx['C'] = current_order.index('C')
    new_ref_pt = ref_pt[ [swap_idx[ new_order[0] ]
                        ,swap_idx[ new_order[1] ]
                        ,swap_idx[ new_order[2] ]
                        ,swap_idx[ new_order[3] ]] ]

    return new_ref_pt


# Mouse-selected
@magicgui(call_button='Transform image')
def transform_image(
    reference_image: Image,
    image2transform: Image,
    axes_order: str = 'ZYX',
    delta_x: float=0.0,
    move_left: bool=False,
    delta_y: float=0.0,
    move_up: bool=True,
    rotate_theta:float=0.0,
    rotate_left:bool=False,
    ) -> Image:

    '''
    Perform 3D rigid-body transformations given an input image and manually set transformation parameters
    translate z/y/x: pixel-wise translations
    rotate_theta: rotation angle in degrees

    Optionally, select a second channel to also transform with the same transformation matrix

    Output will be the transformed image(s), with _transformed appended to name(s)
    '''

    # Grab the image data + convert deg->radians
    image_data = image2transform.data.astype(float)
    reference_image_data = reference_image.data
    rotate_theta = np.deg2rad(rotate_theta)
    assert(len(axes_order) == image_data.ndim)

    # Grab the reference point layers
    ref_point_name = reference_image.name + '_ref_point'
    if ref_point_name in [l.name for l in viewer.layers]:
        ref_point = viewer.layers[ref_point_name].data[0]

    moving_point_name = image2transform.name + '_ref_point'
    if moving_point_name in [l.name for l in viewer.layers]:
        moving_point = viewer.layers[moving_point_name].data[0]

    if image_data.ndim == 3:
        assert(len(moving_point) == 3)
    elif image_data.ndim == 4:
        # Make CZYX
        image_data = swap_axes_order(image_data,axes_order,'CZYX')
        reference_image_data = swap_axes_order(reference_image_data,axes_order,'CZYX')
        ref_point = swap_axes_order_ref_point(ref_point,axes_order,'CZYX')[1:]
        moving_point = swap_axes_order_ref_point(moving_point,axes_order,'CZYX')[1:]

    reference_z,reference_y,reference_x = ref_point.astype(int)
    moving_z,moving_y,moving_x = moving_point.astype(int)

    # handle the directions
    if move_left:
        x_sign = 1
    else:
        x_sign = -1
    if move_up:
        y_sign = 1
    else:
        y_sign = -1
    if rotate_left:
        rot_sign = 1
    else:
        rot_sign = -1

    # # Apply to first image
    array = np.zeros_like(image_data)
    # xy transformations (do slice by slice)
    Txy = EuclideanTransform(translation=[moving_x-reference_x + x_sign*delta_x,
                                          moving_y-reference_y + y_sign*delta_y], rotation=rotate_theta*rot_sign)
    if image_data.ndim == 3:
        for z,im in enumerate(image_data):
            array[z,...] = warp(im, Txy)
        array = z_translate_and_pad(reference_image_data,array,reference_z,moving_z)
        array = array.astype(np.uint16)
    elif image_data.ndim == 4:
        for c,stack in enumerate(image_data):
            for z,im in enumerate(stack):
                array[c,z,...] = warp(im, Txy)
            array[c,...] = z_translate_and_pad(reference_image_data[c,...],array[c,...],reference_z,moving_z)
        array = array.astype(np.uint16)
        array = swap_axes_order(array,'CZYX', axes_order)

    transformed_image = Image(array, name=image2transform.name+'_transformed', blending='additive', colormap=image2transform.colormap)

    return transformed_image


@magicgui(call_button='Filter image')
def filter_gaussian3d( image2filter:Image,
    sigma_xy : float,
    sigma_z : float) -> Image:
    '''
    Perform 3D gaussian filter operation on the selected image using scipy.ndimage.gaussain_filter
    sigma_z: assumed to be first dimension
    sigma_xy: assumed to be second and third dimensions

    Returns a filtered image with _filt appended to name
    '''
    image_data = image2filter.data
    image_data = ndimage.gaussian_filter(image_data,sigma=[sigma_z,sigma_xy,sigma_xy])
    image_data = util.img_as_uint(image_data/image_data.max())
    return Image(image_data,name = image2filter.name + '_filt', blending='additive',colormap=image2filter.colormap)

viewer = napari.Viewer()

viewer.window.add_dock_widget(load_ome_tiffs_and_save_as_tiff,area='left')
viewer.window.add_dock_widget(auto_register_b_and_rshg(),area='left')

viewer.window.add_dock_widget(LoadTimepointForInspection(viewer),area='right')
viewer.window.add_dock_widget(transform_image, area='right')
viewer.window.add_dock_widget(filter_gaussian3d, area='right')

# if __name__ == '__main__':
napari.run()
