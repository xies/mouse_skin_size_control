#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 12:30:54 2022

@author: xies
"""

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from skimage import draw, filters, measure


def fill_in_cube(img,coordinates,label,size=5):
    '''
    Takes in a 3D image and 'fill' a cube with given size at the given
    coordinate and given label
    '''
    assert(img.ndim == 3)
    assert(img.ndim == len(coordinates))

    [z,y,x] = coordinates
    ZZ,YY,XX = img.shape
    lower_x = max(x - size,0)
    lower_y = max(y - size,0)
    lower_z = max(z - size//2,0)
    higher_x = min(x + size,XX)
    higher_y = min(y + size,YY)
    higher_z = min(z + size//2,ZZ)
    img[lower_z:higher_z,lower_y:higher_y,lower_x:higher_x] = label

    return img
    


def draw_labels_on_image(coords,labels,im_shape,font_size=10,fill='white'):
    
    if fill == 'random':
        import random
        fill = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]

    
    # im = np.zeros(im_shape)
    image = Image.new('L',im_shape)
    
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/System/Library/Fonts/ArialHB.ttc',font_size)
    
    for i,l in enumerate(labels):
        text = f'{l}'
        draw.text([coords[i,1],coords[i,0]],text,font=font,fill=fill)
    
    return image


def draw_adjmat_on_image(A,vert_coords,im_shape):
    # assert 2d coords!!
    assert(vert_coords.shape[1] == 2)
    im = np.zeros(im_shape)
    num_verts = A.shape[0]
    
    # To avoid double drawing, trim to upper triangle
    # NB: diagonal should be 0
    # A = np.triu(A)
    
    for idx in range(num_verts):
        this_coord = np.round(vert_coords[idx,...]).astype(int)
        neighbor_idx = np.where(A[idx,:])[0]
        
        for neighbor in neighbor_idx:
            neighbor_coord = np.round(vert_coords[neighbor,...]).astype(int)
            # print(neighbor_coord)
            rr,cc = draw.line(this_coord[0],this_coord[1],neighbor_coord[0],neighbor_coord[1])
            im[rr,cc] = idx+1
            
    return im
        

def draw_adjmat_on_image_3d(A,vert_coords,im_shape):
    # assert 2d coords!!
    assert(vert_coords.shape[1] == 3)
    im = np.zeros(im_shape)
    num_verts = A.shape[0]
    
    # To avoid double drawing, trim to upper triangle
    # NB: diagonal should be 0
    A = np.triu(A)
    
    for idx in range(num_verts):
        this_coord = np.round(vert_coords[idx,...]).astype(int)
        neighbor_idx = np.where(A[idx])[0]
        
        for neighbor in neighbor_idx:
            neighbor_coord = np.round(vert_coords[neighbor,...]).astype(int)
            # print(neighbor_coord)
            lin = draw.line_nd(this_coord,neighbor_coord)
            im[lin] = idx + 1
    return im


def most_likely_label(labeled,im,pixel_threshold=100):
    '''
    For use as an property function to give to skimage.measure.regionprops
    
    Given a mask image, return the intensity value within that image that the highest occurence
    
    
    '''
    label = 0
    if len(im[im>0]) > 0:
        unique,counts = np.unique(im[im > 0],return_counts=True)
        label = unique[counts.argmax()]
        if counts.max() < pixel_threshold:
            label = np.nan
        if label == 0:
            label = np.nan
    return label


def colorize_segmentation(seg,value_dict,dtype=int):
    '''
    Given a segmentation label image, colorize the segmented labels using a dictionary of label: value
    '''
    
    assert( len(np.unique(seg[1:]) == len(value_dict)) )
    colorized = np.zeros_like(seg,dtype=dtype)
    for k,v in value_dict.items():
        colorized[seg == k] = v
    return colorized

