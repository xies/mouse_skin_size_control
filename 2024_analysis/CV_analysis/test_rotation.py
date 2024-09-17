#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:03:41 2024

@author: xies

"""
from skimage.transform import EuclideanTransform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

A = pd.read_csv('/users/xies/Desktop/Registered_G_ref_A.csv',index_col=0)
A = A.rename(columns={'axis-0':'T','axis-1':'Z','axis-2':'Y','axis-3':'X'})
B = pd.read_csv('/users/xies/Desktop/Registered_G_ref_B.csv',index_col=0)
B = B.rename(columns={'axis-0':'T','axis-1':'Z','axis-2':'Y','axis-3':'X'})

#%%

reference_index = 0

C1 = np.squeeze(np.stack([A[A['T'] == reference_index][['Y','X']], B[B['T'] == reference_index][['Y','X']]]))
C2 = np.squeeze(np.stack([A[A['T'] == 1][['Y','X']], B[B['T'] == 1][['Y','X']]]))


plt.subplot(2,1,1)

plt.scatter(C1[:,0],C1[:,1],color='r')
plt.scatter(C2[:,0],C2[:,1],color='m',marker='+')

com1,com2,Rest = estimate_translation_and_rotation_from_anchors(C1,C2)

C2T = (Rest @ C2.T).T
C2f = C2T - (com2[None,:] - com1[None,:])

plt.subplot(2,1,2)
plt.scatter(C1[:,0],C1[:,1],color='r')
plt.scatter(C2T[:,0],C2T[:,1],color='m',marker='+')
plt.scatter(C2f[:,0],C2f[:,1],color='m',marker='*')

#%%

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

#%%



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
    reference_z = anchors_A[reference_index,...][1]
    output_stack = np.zeros_like(image_data)
    for t,anchor_A in enumerate(anchors_A):
        
        # Grab current image
        this_im = image_data[t,...]
        
        # IF this is is the refence point, put in original image and skip
        if t == reference_index:
            output_stack[t,...] = util.img_as_uint(this_im/this_im.max())
            continue

        # First, z-translate (always use anchorA for now)
        moving_z = anchors_A[t,...][1]
        array = z_translate_and_pad(reference_image,this_im,reference_z,moving_z)
        
        # Now, calculate translation and rotation together, disregarding z
        moving_cloud = np.stack([ anchors_A[t,2:4], anchors_B[t,2:4]])
        moving_cloud = np.squeeze(moving_cloud)
        
        com_ref,com_moving,Rm = estimate_translation_and_rotation_from_anchors(reference_cloud,moving_cloud)
        
        dy,dx = com_ref-com_moving
        
        Txy = EuclideanTransform(translation=[dy,dx],rotation=Rm)
        array = np.zeros_like(this_im)
        for z,im in enumerate(this_im):
            array[z,...] = warp(im,Txy)
        
        output_stack[t,...] = util.img_as_uint(array/array.max())
        
    return( Image(output_stack,name='Aligned') )



    
    