#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:53:16 2019

@author: xies
"""

import numpy as np
from glob import glob
import pandas as pd
import seaborn as sb
from os import path

dirname = '/Users/xies/Box/Mouse/Skin/Mesa et al/W-R2/Stem v diff/'

cell_dicts = {}

for f in glob(path.join(dirname,'*.txt')):
    print f
    cell = pd.read_csv(f,sep='\t')
    
    # Parse cell type, channel etc
    basename = path.basename(f)
    basesplit = basename.split('.')
    cellID = basesplit[0]
    cell_type = basesplit[1]
    channel = basesplit[2]
    
    channel_total = cell.RawIntDen.sum()
    this_cell_dict = {channel:float(channel_total)}
    volume = cell.Area.sum()
    this_cell_dict['volume'] = float(volume)
    this_cell_dict['type'] = cell_type
    
    # Find the cellID this belongs to and append the current channel keypair
    # Check if key already exist
    if cellID in cell_dicts.keys():
        cell_dicts[cellID] = merge_two_dicts(cell_dicts[cellID],this_cell_dict)
    else:
        cell_dicts[cellID] = this_cell_dict
        
# Flatten out the dict of dict
tmp_ = []
for cellID,data in cell_dicts.iteritems():
    tmp_.append([ cellID,data['type'],data['h2b'],data['actin'],data['fucci'],data['volume'],
                 data['h2b']/data['volume'],data['actin']/data['volume'],data['fucci']/data['volume'],
                 data['fucci']/data['h2b'],data['fucci']/data['actin'],np.log(data['fucci']/data['volume']) ] )

df = pd.DataFrame(tmp_,columns=['CellID','Type','H2B','Actin','FUCCI','Volume',
                                'H2B concentration','Actin concentration','FUCCI concentration',
                                'FUCCI/H2b','FUCCI/actin','FUCCI log'])
df.index = df.CellID

# Calculate all pairwise ratios

sb.catplot(data=df,x='Type',y='Volume')
sb.catplot(data=df,x='Type',y='H2B concentration')
sb.catplot(data=df,x='Type',y='FUCCI concentration')
sb.catplot(data=df,x='Type',y='Actin concentration')

sb.catplot(data=df,x='Type',y='FUCCI log')

stem = df[df.Type == 'stem']
spinous = df[df.Type == 'spinous']
granular = df[df.Type == 'granular']

plt.figure()
plt.scatter(stem.Volume,stem['FUCCI'])
plt.scatter(spinous.Volume,spinous['FUCCI'])
plt.scatter(granular.Volume,granular['FUCCI'])
plt.legend(['stem','spinous','granular'])

plt.xlabel('Nuclear volume')
plt.ylabel('FUCCI-G1')



