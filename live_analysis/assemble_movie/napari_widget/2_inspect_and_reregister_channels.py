from pathlib import Path
from itertools import cycle
from typing import List

from magicgui import magicgui
import napari
from napari.types import ImageData
from napari.layers import Image
from napari.utils.notifications import show_warning

import numpy as np
from skimage import io
from skimage.transform import EuclideanTransform, warp
from scipy import ndimage

from twophotonUtils import parse_unregistered_channels, parse_unaligned_channels

# Default choices for the timepoints to insepct
DEFAULT_CHOICES = [np.nan]

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
        widget.dropdown.choices = choices

def load_images():
    '''
    Defines a image loader widget that auto-populates a list of files
    
    '''
    @magicgui(dirname={'label': 'Image region to load:','mode': 'd'}, # Restrict file picker to directories
        dropdown={'choices':DEFAULT_CHOICES, 'label':'Region to load:'}, # Begins with a default NaN choice
        pre_registered={'label':'Load pre-registered images?'}, # option to load pre-registered or unregistered images
        call_button='Load timepoint')
    def widget(viewer: napari.Viewer, dropdown, dirname=Path.home(),pre_registered=False):
        if pre_registered:
            filelist = parse_unaligned_channels(dirname) #@todo: auto-detect when these are not yet available
        else:
            filelist = parse_unregistered_channels(dirname)
        # Load the files using a cycling colormap series
        ind_to_load = dropdown
        file_tuple = filelist.iloc[ind_to_load]
        colormaps = cycle(['bop blue','gray','bop orange','bop purple'])
        for name,filename in file_tuple.items():
            viewer.add_image(io.imread(filename),name=name, blending='additive', colormap=next(colormaps))

    @widget.dirname.changed.connect
    def update_timepoints_on_file_change(event=None):
        _update_timepoints_on_file_change(widget)
    return widget

@magicgui(call_button='Transform image')
def transform_image(
    reference_image: Image,
    image2transform: Image,
    second_channel: Image, 
    translate_x: int=-500, #@todo: unclear why magicgui doesn't let you go more negative than the default number
    translate_y: int=-500,
    translate_z: int=-100,
    rotate_theta:float=0.0,
    Transform_second_channel:bool=False,
    Truncate_in_z:bool=True
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

    # z-transformations (up an down stack)... use the form EuclideanTransform[\0,-dz] can't figure out why @todo
    Tz = EuclideanTransform(translation=[0,-translate_z],dimensionality=2)
    print()
    # xy transformations (do slice by slice)
    Txy = EuclideanTransform(translation=[-translate_x,-translate_y], rotation=rotate_theta)

    # Apply to first image
    array = warp(image_data,Tz)
    for z,im in enumerate(array):
        array[z,...] = warp(im, Txy)
    if Truncate_in_z:
        array = array[:reference_image.data.shape[0],...]
        
    transformed_image = Image(array, name=image2transform.name+'_transformed', blending='additive', colormap=image2transform.colormap)
    output_list = [transformed_image]
    
    if Transform_second_channel:
        array = warp(second_image_data,Tz)
        for z,im in enumerate(array):
            array[z,...] = warp(im, Txy)
            array = array[:reference_image.data.shape[0],...]
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
viewer.window.add_dock_widget(load_images(),area='right')
viewer.window.add_dock_widget(transform_image, area='right')
viewer.window.add_dock_widget(filter_gaussian3d, area='right')

if __name__ == '__main__':
    napari.run()
