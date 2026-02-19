#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:59:29 2022

@author: xies
"""

from glob import glob
from natsort import natsorted
import pandas as pd
from re import findall
from os import path
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from tqdm import tqdm

import xml.etree.ElementTree as ET
from dateutil import parser
from collections import OrderedDict

def return_prefix(filename):

    # Use a function to regex the Day number and use that to sort
    day = findall('.(\d+)\. ',filename)
    assert(len(day) == 1)

    return int(day[0])

def align_timecourse_by_curvature(region_dir,
                                  channel_names,
                                  spacing,
                                  reference_t:int = 0,
                                  decimation_factor:int=30,
                                  substr:str='*. Day*',
                                  overwrite:bool=False):

    from measurements import get_tissue_curvature_over_grid, get_mesh_from_bm_image
    from pystackreg import StackReg
    from skimage import transform, util

    day_directories = natsorted(glob(path.join(region_dir,substr)))
    TT = len(day_directories)

    Xtrim = slice(200,800)
    Ytrim = slice(200,800)

    # Calculate mesh + curvature
    images = {}
    curvature = np.zeros((TT,1024,1024))
    heightmaps = {}
    for t,day in enumerate(day_directories):
        images[t] = {channel: io.imread(path.join(day,f'{channel}_reg.tif')) for channel in channel_names}
        heightmap = io.imread(path.join(day,'heightmap.tif'))
        heightmaps[t] = heightmap
        if path.exists(path.join(day,'mean_curvature.npz')) and not overwrite:
            curvature[t,...] = np.load(path.join(day,'mean_curvature.npz'))['mean_curvature']

        else:
            height_image = io.imread(path.join(day,'height_image.tif'))
            mesh = get_mesh_from_bm_image(height_image, spacing=spacing, decimation_factor=decimation_factor)
            curvature[t,...],_ = get_tissue_curvature_over_grid(mesh,
                                                         image_shape=height_image.shape)
            np.savez(path.join(day,'mean_curvature.npz'),mean_curvature=curvature[t,...])

    # Stackreg
    # Reg to a fixed frame (not rolling)
    if path.exists(path.join(region_dir,'Tmats.npz')) and not overwrite:
        Tmats = np.load(path.join(region_dir,'Tmats.npz'))['Tmats']
    else:
        transformed_curvatures = np.zeros_like(curvature)
        transformed_curvatures[0,...] = curvature[0,...]
        Tmats = np.zeros((TT,3,3))
        Tmats[reference_t,...] = np.eye(3)

        target = transformed_curvatures[reference_t,Ytrim,Xtrim]

        for t,day in enumerate(set(np.arange(TT)) - set([reference_t]) ):

            moving = curvature[t][Ytrim,Xtrim]

            sr = StackReg(StackReg.RIGID_BODY)
            T = sr.register(target,moving)
            T = transform.EuclideanTransform(T)
            Tmats[t+1,...] = T
            print(T)
            transformed_curvatures[t,...] = transform.warp(curvature[t],T)

        np.savez(path.join(region_dir,'Tmats.npz'),Tmats = Tmats)


    # Transform heightmap to get the z-alignment
    mean_heightmap = \
        [transform.warp(heightmaps[t],Tmats[t])[slice(400,500),slice(400,500)].mean().astype(int)
            for t in range(TT)]
    print(mean_heightmap)

    # Transform other channels
    transformed_channels = {}
    for channel in tqdm(channel_names):

        shape = images[0][channel].shape
        this_channel_transformed = []
        for t in range(TT):
            this_channel_this_day = images[t][channel]
            this_channel_this_day_transformed = np.zeros_like(this_channel_this_day)
            print(f'{t}')
            for z,im in enumerate(this_channel_this_day):
                this_channel_this_day_transformed[z,...] = transform.warp(im.astype(float),Tmats[t])
            this_channel_transformed.append(this_channel_this_day_transformed)

        this_channel_transformed = z_align_ragged_timecourse(this_channel_transformed,mean_heightmap)
        this_channel_transformed = util.img_as_uint(this_channel_transformed / this_channel_transformed.max())
        io.imsave(path.join(region_dir,f'{channel}_align.tif'),this_channel_transformed)
        transformed_channels[channel] = this_channel_transformed
        print('Done')

    return transformed_channels,Tmats

def parse_aligned_timecourse_directory(dirname,folder_str='*. */',SPECIAL=[]):
    # Given a directory (of Prairie Instruments time course)
    #

    B = glob(path.join(dirname,folder_str, 'B_align.tif'))
    idx = [return_prefix(f) for f in B]
    filelist = pd.DataFrame(index=idx)
    filelist.loc[idx,'B'] = B

    G = glob(path.join(dirname,folder_str, 'G_align.tif'))
    idx = [return_prefix(f) for f in G]
    filelist.loc[idx,'G'] = G

    R = glob(path.join(dirname,folder_str, 'R_align.tif'))
    idx = [return_prefix(f) for f in R]
    filelist.loc[idx,'R'] = R

    R_shg = glob(path.join(dirname,folder_str, 'R_shg_align.tif'))
    idx = [return_prefix(f) for f in R_shg]
    filelist.loc[idx,'R_shg'] = R

    # T = len(filelist)

    for t in SPECIAL:
        # t= 0 has no '_align'imp
        # s = pd.DataFrame({'B': sorted(glob(path.join(dirname,folder_str, 'B_reg.tif')))[0],
        #                   'G': sorted(glob(path.join(dirname,folder_str, 'G_reg.tif')))[0],
        #                   'R': sorted(glob(path.join(dirname,folder_str, 'R_reg_reg.tif')))[0],
        #               'R_shg': sorted(glob(path.join(dirname,folder_str, 'R_shg_reg_reg.tif')))[0]},
        #                  index=[t])

        B = glob(path.join(dirname,f'{t}. */B_reg.tif'))[0]
        filelist.loc[t,'B'] = B
        G = glob(path.join(dirname,f'{t}. */G_reg.tif'))[0]
        filelist.loc[t,'G'] = G
        R = glob(path.join(dirname,f'{t}. */R_reg_reg.tif'))[0]
        filelist.loc[t,'R'] = R
        R_shg = glob(path.join(dirname,f'{t}. */R_shg_reg_reg.tif'))[0]
        filelist.loc[t,'R_shg'] = R_shg

        # filelist = pd.concat((s,filelist))
    filelist = filelist.sort_index()

    return filelist


def parse_unregistered_channels(dirname,folder_str='*. Day*/',sort_func=return_prefix):
    # Given a directory (of Prairie Instruments time course), grab all the _reg.tifs
    # (channels are not registered to each other)
    #


    B = glob(path.join(dirname,folder_str, 'B_reg.tif'))
    idx = [return_prefix(f) for f in B]
    filelist = pd.DataFrame(index=idx)
    filelist.loc[idx,'B'] = B

    # filelist['G'] = sorted(glob(path.join(dirname,folder_str, 'G_reg.tif')), key = return_prefix)
    G = glob(path.join(dirname,folder_str, 'G_reg.tif'))
    idx = [return_prefix(f) for f in G]
    filelist.loc[idx,'G'] = G

    R = glob(path.join(dirname,folder_str, 'R_reg.tif'))
    idx = [return_prefix(f) for f in R]
    filelist.loc[idx,'R'] = R

    R_shg = glob(path.join(dirname,folder_str, 'R_shg_reg.tif'))
    idx = [return_prefix(f) for f in R_shg]
    filelist.loc[idx,'R_shg'] = R_shg

    filelist = filelist.sort_index()

    return filelist

def parse_unaligned_channels(dirname,folder_str='*. Day*/'):
    # Given a directory (of Prairie Instruments time course)
    #

    # filelist['B'] = sorted(glob(path.join(dirname,folder_str, 'B_reg.tif')), key = return_prefix)
    B = glob(path.join(dirname,folder_str, 'B_reg.tif'))
    idx = [return_prefix(f) for f in B]
    filelist = pd.DataFrame(index=idx,columns=['B','G','R','R_shg'])
    filelist.loc[idx,'B'] = B

    # filelist['G'] = sorted(glob(path.join(dirname,folder_str, 'G_reg.tif')), key = return_prefix)
    G = glob(path.join(dirname,folder_str, 'G_reg.tif'))
    idx = [return_prefix(f) for f in G]
    filelist.loc[idx,'G'] = G

    R = glob(path.join(dirname,folder_str, 'R_reg_reg.tif'))
    idx = [return_prefix(f) for f in R]
    filelist.loc[idx,'R'] = R

    R_shg = glob(path.join(dirname,folder_str, 'R_shg_reg_reg.tif'))
    idx = [return_prefix(f) for f in R_shg]
    filelist.loc[idx,'R_shg'] = R_shg

    filelist = filelist.sort_index()
    # T = len(filelist)

    return filelist

def plot_cell_volume(track,x='Frame',y='Volume'):
    t = track[x]
    y = track[y]
    if 'Mitosis' in track.columns:
        if track.iloc[0]['Mitosis']:
            t = t[:-1]
            y = y[:-1]
    plt.plot(t,y)


def parse_XML_timestamps(region_dir,subdir_str='*. Day*/',beginning=0):

    T = len(glob(path.join(region_dir,subdir_str)))
    timestamps = OrderedDict()

    for t in range(beginning,beginning+T):

        subfolders = glob(path.join(region_dir,f'{t}.*/ZSeries*/'))

        for d in subfolders:
            ome_tifs = glob(path.join(d,'*.ome.tif'))
            xmls = glob(path.join(d,'*.xml'))
            if len(ome_tifs) > 40:
                if len(findall('1020nm',path.split(path.split(d)[0])[1])) == 0:

                    tree = ET.parse(xmls[0])
                    timestr = tree.getroot().attrib['date']
                    timestamp = parser.parse(timestr)

        timestamps[t] = timestamp

    return timestamps


def parse_voxel_resolution_from_XML(region_dir):

    xmls = glob(path.join(region_dir,'*.*/ZSeries*/*.xml'))

    tree = ET.parse(xmls[0])
    root = tree.getroot()

    for child in root:
        if child.tag == 'PVStateShard':
            PVState = child
            break

    for child in PVState:
        if child.attrib['key'] == 'micronsPerPixel':
            microns_resolution = child

    for child in microns_resolution:
        if child.attrib['index'] == 'XAxis':
            dx = child.attrib['value']
        if child.attrib['index'] == 'ZAxis':
            dz = child.attrib['value']

    return float(dx),float(dz)

def z_align_ragged_timecourse(ragged_stack_list,same_Zs):
    '''
    Takes two z-stacks and translate the im_moving so that z_ref and z_moving will end up
    being the same index in the translated image. Will also truncate/pad im_moving
    to be the same size as im_ref

    '''
    same_Zs = same_Zs.astype(int)
    original_stack_sizes = np.array([x.shape[0] for x in ragged_stack_list])
    top_size = (original_stack_sizes - same_Zs)

    ragged_bottom_Z = same_Zs.max() - same_Zs

    TT = len(ragged_stack_list)
    XX = ragged_stack_list[0].shape[1]

    aligned_stack_timecourse = np.zeros((TT,int(same_Zs.max()+top_size.max()),XX,XX))

    for t in tqdm(range(TT)):

        # Take the 'bottom' portion of original stack
        aligned_stack_timecourse[t, ragged_bottom_Z[t]:same_Zs.max() ,:,:] \
            = ragged_stack_list[t][0:same_Zs[t],:,:]
        # Take the 'top' portion of the original stack
        aligned_stack_timecourse[t,same_Zs.max():same_Zs.max()+top_size[t],:,:] \
            = ragged_stack_list[t][same_Zs[t]:,:,:]

    return aligned_stack_timecourse
