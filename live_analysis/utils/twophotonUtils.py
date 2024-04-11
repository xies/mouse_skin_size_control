#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:59:29 2022

@author: xies
"""

from glob import glob
import pandas as pd
from re import findall
from os import path
import matplotlib.pyplot as plt
import numpy as np


import xml.etree.ElementTree as ET
from dateutil import parser
from collections import OrderedDict

def return_prefix(filename):

    # Use a function to regex the Day number and use that to sort
    day = findall('.(\d+)\. ',filename)
    assert(len(day) == 1)

    return int(day[0])

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
    filelist = pd.DataFrame(index=idx)
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
