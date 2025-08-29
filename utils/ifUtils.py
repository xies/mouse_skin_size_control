#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 18:10:51 2019

@author: mimi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from SelectFromCollection import SelectFromCollection

class Gate:
    def __init__(self,name,xfield,yfield):
        self.name = name
        self.xfield = xfield
        self.yfield = yfield
        self.selector: None
        self.path = None

    def draw_gates(self,df,alpha=0.01):
        xgate = self.xfield
        ygate = self.yfield
        plt.figure()
        pts = plt.scatter(df[xgate],df[ygate], alpha=alpha)
        plt.xlabel(xgate)
        plt.ylabel(ygate)
        self.selector = SelectFromCollection(plt.gca(), pts)

    def get_gated_indices(self,df):
        verts = np.array(self.selector.poly.verts)
        x = verts[:,0];y = verts[:,1]
        p_ = Path(np.array([x,y]).T)
        self.path = p_
        I = np.array([p_.contains_point([x,y]) for x,y in zip(df[self.xfield],df[self.yfield])])
        return I

    def draw_gate_as_patch(self,df,alpha=0.01):
        plt.figure()
        plt.scatter(df[self.xfield],df[self.yfield],alpha=alpha)
        plt.xlabel(self.xfield);plt.ylabel(self.yfield);
        patch = PathPatch(self.path,lw=2,facecolor='r',alpha=0.2)
        plt.gca().add_patch(patch)

def min_normalize_image(im):
    im = im - im.min()

    return im

def subtract_nonmask_background(img,bg_mask,erosion_radius=5,mean=np.mean):
    from skimage import morphology
    """

    Use the 'ON' pixels from a binary mask of BACKGROUND to estimate background
    intensity. Uses an disk of radius 5 (default) to 'erode' first. Background
    intensity is defined as the mean of the leftover background pixels.

    """
    import numpy as np
    disk = morphology.selem.disk(erosion_radius) # for bg subtraction on RB channel
    bg_pixels = morphology.erosion(bg_mask,disk)
    bg = np.median(img[bg_pixels])
    img_sub = img.copy()
    img_sub = img_sub - bg
    img_sub[img_sub < 0] = 0
    return img_sub


def delete_border_objects(labels):
    import numpy as np
    """

    Givenb a bwlabeled image, return the labels of the objects that touch the image border (1px wide)
    Background object should be labeled 0

    """
    if labels.ndim == 2:
        border = np.zeros(labels.shape)
        border[0,:] = 1
        border[-1,:] = 1
        border[:,0] = 1
        border[:,-1] = 1
        touch_border = np.unique(labels[border == 1])
    elif labels.ndim == 3:
        border = np.zeros(labels.shape)
        border[0,:,:] = 1
        border[-1,:,:] = 1
        border[:,0,:] = 1
        border[:,-1,:] = 1
        border[:,:,1] = 1
        border[:,:,1] = 1
        touch_border = np.unique(labels[border == 1])

    touch_border[touch_border > 0]
    for touching in touch_border:
        labels[labels == touching] = 0

    return labels # Get rid of 'background object'

def subtract_cytoplasmic_ring(img,nuclear_mask,inner_r=3,outer_r=5):
    from skimage import morphology
    import numpy as np

    """
    Generate a 'ring' of background pixels around the nuclei in an image.
    inner_r -- inner ring radius, # of pixels dilated from mask (default = 3)
    outer_r -- outer ring radius (default = 5)

    """

    kernel = morphology.disk(inner_r)
    inner_mask = morphology.dilation(nuclear_mask,kernel)
    kernel = morphology.disk(outer_r)
    outer_mask = morphology.dilation(nuclear_mask,kernel)
    ring = np.logical_xor(outer_mask,inner_mask)

    bg = np.mean(img[ring])
    img_sub = img.copy() - bg
    img_sub[img_sub < 0] = 0

    return img_sub
