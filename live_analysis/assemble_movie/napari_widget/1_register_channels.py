#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:42:36 2024

@author: xies
"""

from pathlib import Path
from itertools import cycle
from typing import List

from magicgui import magicgui
import napari
from napari.layers import Image
from napari.utils.notifications import show_warning
from napari.utils import progress

# from os import path
import numpy as np
from skimage import io,util,filters
from skimage.transform import EuclideanTransform, warp
from scipy import ndimage

from pystackreg import StackReg
from twophotonUtils import parse_unregistered_channels, parse_unaligned_channels, find_most_likely_z_slice_using_CC, \
    z_translate_and_pad

from os import path
from glob import glob
from re import findall

# Default choices for the timepoints to insepct
global DEFAULT_CHOICES
DEFAULT_CHOICES = [0,1,2]

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
    OVERWRITE: bool=False
    ):
    
    subfolders = glob(path.join(dirname,pattern_str))
    print(subfolders)
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
    print(header_ome_h2b)
    print(header_ome_fucci)

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
        
        # Use StackReg
        print(f'Registering {d}')
        sr = StackReg(StackReg.TRANSLATION) # There should only be slight sliding motion within a single stack
        T = sr.register_stack(B,reference='previous',n_frames=20,axis=0) #Obtain the transformation matrices
        B_reg = sr.transform_stack(B,tmats=T) # Apply to both channels
        G_reg = sr.transform_stack(G,tmats=T)
        
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
        print(f'Registering {d}')
        sr = StackReg(StackReg.TRANSLATION) # There should only be slight sliding motion within a single stack
        T = sr.register_stack(R,reference='previous',axis=0) #Obtain the transformation matrices
        R_reg = sr.transform_stack(R,tmats=T) # Apply to both channels
        R_shg_reg = sr.transform_stack(R_shg,tmats=T) # Apply to both channels
        
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
    choices = None
    filelist = parse_unregistered_channels(dirname)
    
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
        timepoints_to_register=(0),
        OVERWRITE: bool=False,
        ):
    
        filelist = parse_unregistered_channels(dirname)
        
        for t in progress(timepoints_to_register):
        
            # Check for overwriting
            output_dir = path.split(path.dirname(filelist.loc[t,'R']))[0]
            if path.exists(path.join(path.dirname(filelist.loc[t,'R']),'R_reg_reg.tif'))  and not OVERWRITE:
            # and path.exists(path.join(path.dirname(B_tifs[t]),'B_reg_reg.tif'))  and not OVERWRITE:
                print(f'Skipping t = {t} because its R_reg_reg.tif already exists')
                continue
                
            print(f'\n--- Started t = {t} ---')
            B = io.imread(filelist.loc[t,'B'])
            R_shg = io.imread(filelist.loc[t,'R_shg'])
            # G = io.imread(filelist.loc[t,'G'])
            R = io.imread(filelist.loc[t,'R'])    
            print('Done reading images')
                
            # Find the slice with maximum mean value in R_shg channel
            z_ref = R_shg.mean(axis=2).mean(axis=1).argmax()
            print(f't = {t}: R_shg max std at {z_ref}')
            R_ref = R_shg[z_ref,...]
            R_ref = filters.gaussian(R_ref,sigma=0.5)
            z_moving = find_most_likely_z_slice_using_CC(R_ref,B)
            print(f'Cross correlation done and target Z-slice set at: {z_ref}')
            target = filters.gaussian(B[z_ref,...],sigma=0.5)
        
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
            R_padded = z_translate_and_pad(B,R_transformed,z_ref,z_moving)
            R_shg_padded = z_translate_and_pad(B,R_shg_transformed,z_ref,z_moving)
                
            output_dir = path.dirname(filelist.loc[t,'G'])
        
            print('Saving')
            io.imsave(path.join(output_dir,'R_reg_reg.tif'),util.img_as_uint(R_padded/R_padded.max()),check_contrast=False)
            io.imsave(path.join(output_dir,'R_shg_reg_reg.tif'),util.img_as_uint(R_shg_padded/R_shg_padded.max()),check_contrast=False)
        
    @widget.dirname.changed.connect
    def update_timepoints_on_file_change(event=None):
        _update_timepoints_on_file_change(widget)
    return widget

'''
Step three: Load, inspect, and manually fix registration
'''

from magicgui.widgets import ComboBox, Container, FileEdit, PushButton

class LoadDirectoryForInspection(Container):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dropdown_choices = ()
        self.append(
            FileEdit(name="dirname",label="Image region to load:",mode="d")
        )
        self.append(
            ComboBox(name="timepoint2load", choices=self.get_dropdown_choices, label="Select timepoint")
        )
        self.append(PushButton(name="load_button", text="Load timepoint"))
        self.append()

        self.dirname.changed.connect(self.update_dropdown_choices)
        self.load_button.changed.connect(self.load_images)

    def get_dropdown_choices(self, dropdown_widget): #default
        return self._dropdown_choices
    
    def update_dropdown_choices(self):
        dirname = self.dirname.value
        choices = None
        filelist = parse_unregistered_channels(dirname)
        if len(filelist) > 0:
            choices = filelist.index.values
        else:
            show_warning(f'Directory {dirname} is not a region directory.')
        if choices is not None:
            print(choices)
            self._dropdown_choices = choices
            self.timepoint2load.reset_choices()
            
    def load_images(self) -> List[napari.layers.Layer]:
        
        if pre_registered:
            filelist = parse_unaligned_channels(self.dirname.value) #@todo: auto-detect when these are not yet available
            suffix = '_reg_reg'
        else:
            filelist = parse_unregistered_channels(self.dirname.value)
            suffix = '_reg'
        # Load the files using a cycling colormap series
        file_tuple = filelist.loc[self.timepoint2load.value[0]]
        colormaps = cycle(['bop blue','gray','bop orange','bop purple'])
        layers_list = []
        for name,filename in file_tuple.items():
            layers_list.append(Image(io.imread(filename),name=name+suffix, blending='additive', colormap=next(colormaps)))
            
        return layers_list

# def get_available_timepoints(widget):
#     if not hasattr(widget,'_timepoint_choices'):
#         widget._timepoint_choices = ()
#     return widget._timepoint_choices
    
# def _update_inspection_timepoint_on_file_change(widget):
#     """
#     Called whenever the file picker is changed. Will look into that directory,
#     and return the available timepoints as defined by parse_unregistered_channels
#     """
    
#     dirname = widget.dirname.value
#     choices = None
#     filelist = parse_unregistered_channels(dirname)
#     if len(filelist) > 0:
#         choices = filelist.index.values
#     else:
#         choices = ()
#         show_warning(f'Directory {dirname} is not a region directory.')
#     widget._timepoint_choices = choices
#     # widget.timepoint2load.choices = choices
#     return choices
    
# def load_images_for_inspection():
#     '''
#     Image loader widget that auto-populates a list of files
#     @todo: Currentlly buggy with the following behavior: when an image is added,
#     all parameters will reset to default.
#     https://forum.image.sc/t/dynamic-select-options-reset/81159/5
    
#     '''
    
#     @magicgui(dirname={'label': 'Image region to load:','mode': 'd'}, # Restrict file picker to directories
#         timepoint2load={'widget_type':'Select',
#                   'choices':get_available_timepoints,
#                   'allow_multiple':False,
#                   'label':'Region to load:'}, # Begins with a default NaN choice
#         pre_registered={'label':'Load pre-registered images?'}, # option to load pre-registered or unregistered images
#         call_button='Load timepoint')
#     def widget(dirname=Path.home(),timepoint2load=None,pre_registered=False) -> List[napari.layers.Layer]:
        
#         if pre_registered:
#             filelist = parse_unaligned_channels(dirname) #@todo: auto-detect when these are not yet available
#             suffix = '_reg_reg'
#         else:
#             filelist = parse_unregistered_channels(dirname)
#             suffix = '_reg'
#         # Load the files using a cycling colormap series
#         file_tuple = filelist.loc[timepoint2load[0]]
#         colormaps = cycle(['bop blue','gray','bop orange','bop purple'])
#         layers_list = []
#         for name,filename in file_tuple.items():
#             layers_list.append(Image(io.imread(filename),name=name+suffix, blending='additive', colormap=next(colormaps)))
            
#         return layers_list
        
#     @widget.dirname.changed.connect
#     def update_timepoints_on_file_change(event=None):
#         _update_inspection_timepoint_on_file_change(widget)
#         widget.reset_choices()
        
#     return widget

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
    return Image(image_data,name = image2filter.name + '_filt', blending='additive',colormap=image2filter.colormap)

viewer = napari.Viewer()

# viewer.window.add_dock_widget(load_ome_tiffs_and_save_as_tiff,area='left')
# viewer.window.add_dock_widget(auto_register_b_and_rshg(),area='left')

viewer.window.add_dock_widget(LoadDirectoryForInspection(),area='right')
viewer.window.add_dock_widget(transform_image, area='right')
viewer.window.add_dock_widget(filter_gaussian3d, area='right')

if __name__ == '__main__':
    napari.run()
