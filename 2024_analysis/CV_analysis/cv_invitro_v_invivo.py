#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:44:03 2024

@author: xies
"""

# from parseZ2 import parse_Z2
from glob import glob
import numpy as np
import pandas as pd
from os import path
import matplotlib.pyplot as plt
from z2Parser import parse_Z2

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/Shuyuan Coulter counter data/'

#%% Load Z2s

filelist = glob(path.join(dirname,'*.=#Z2'))[1:]

sizes = {}
for f in filelist:
    
    name = path.splitext( path.split(f)[1] )[0]
    print(name)
    df = parse_Z2(f)
    # Convert 2 volume
    df['Volume'] = (df['Diam']/2)**3 * 4 * np.pi / 3
    sizes[name] = df
    
#%% Individually gate the cutoff for each cell type

for i in range(len(filelist)):
    plt.subplot( 3, int(np.ceil(len(filelist) / 3)), i+1)
    df = sizes[ list(sizes.keys())[i] ]
    plt.plot(df['Volume'],df['Count']/df.Count.sum())
    plt.title( list(sizes.keys())[i] )

lower_cutoff = {'RPE_DMSO 1': 11,
              'HMECs_WT_DMSO':11,
              'Hepato_invitro_WT_P5':11,
              'RPE1':12,
              'NIH3T3':12,
              'mESC':9.5,
              'SX_hESCs':12}


high_cutoff = {'RPE_DMSO 1': 30.5,
              'HMECs_WT_DMSO':26.8,
              'Hepato_invitro_WT_P5':21.25,
              'RPE1':24.25,
              'NIH3T3':24.4,
              'mESC':24.25,
              'SX_hESCs':23.75
              }

#%%

for name,df in sizes.items():
    df = df[ (df['Diam'] > lower_cutoff[name]) & (df['Diam'] < high_cutoff[name]) ]
    sizes[name] = df

def get_mean(df):
    return np.sum(df['Volume']*df['Count']) / df['Count'].sum()

def get_std(df):
    mu = get_mean(df)
    di = np.sum( (df['Volume'] - mu)**2 * df['Count'] )
    return np.sqrt( di / df['Count'].sum() )

def get_cv(df):
    return get_std(df) / get_mean(df)

CVs = {name:get_cv(df) for name,df in sizes.items() }
CVs = pd.DataFrame(CVs,index=['CV']).T
CVs['Category'] = '2D culture'
# CVs['Counts'] = counts
# mean = np.sum(probs * mids)  
# sd = np.sqrt(np.sum(probs * (mids - mean)**2))

CVs.to_excel('/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/cv_invivo_v_invitro.xlsx')

# for i in range(len(filelist)):
#     plt.subplot( 3, len(filelist)//3, i+1)
#     df = sizes[ list(sizes.keys())[i] ]
#     plt.plot(df['Volume'],df['Count'])
#     plt.title( f'{list(sizes.keys())[i]} CV: {CVs_invitro[list(sizes.keys())[i]]}' )


#% Load in vivo / organoids

from skimage import io, measure
import seaborn as sb

# Skin
R1_t0 = io.imread('/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/W-R1/3d_cyto_seg/3d_cyto_manual/t0_cleaned.tif')
R2_t9 = io.imread('/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/W-R2/3d_cyto_seg/3d_cyto_manual/t9_cleaned.tif')

df_R1 = pd.DataFrame(measure.regionprops_table(R1_t0,properties=['area']))
df_R2 = pd.DataFrame(measure.regionprops_table(R2_t9,properties=['area']))

CVs.loc['Skin R1','CV'] = df_R1['area'].std() / df_R1['area'].mean()
CVs.loc['Skin R1','Category'] = 'In vivo skin'
CVs.loc['Skin R1','Counts'] = len(df_R1)
CVs.loc['Skin R2','CV'] = df_R2['area'].std() / df_R2['area'].mean()
CVs.loc['Skin R2','Category'] = 'In vivo skin'
CVs.loc['Skin R2','Counts'] = len(df_R2)

# Organoids
organoid_pos5_t65 = io.imread('/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/manual_segmentation/man_Channel0-T0065.tif')
df_pos5_t65 = pd.DataFrame(measure.regionprops_table(organoid_pos5_t65,properties=['area']))

CVs.loc['organoid_pos5','CV'] = df_pos5_t65['area'].std() / df_pos5_t65['area'].mean()
CVs.loc['organoid_pos5','Category'] = 'Organoid nuclear size'
CVs.loc['organoid_pos5','Counts'] = len(df_pos5_t65)

organoid_pos15_t1 = io.imread('/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 15_2um/nuclei_segmentation/h2birfp670-T0001.tif')
df_pos15_t1 = pd.DataFrame(measure.regionprops_table(organoid_pos15_t1,properties=['area']))

CVs.loc['organoid_pos6','CV'] = df_pos15_t1['area'].std() / df_pos15_t1['area'].mean()
CVs.loc['organoid_pos6','Category'] = 'Organoid nuclear size'
CVs.loc['organoid_pos6','Counts'] = len(df_pos15_t1)

organoid_pos8_t40 = io.imread('/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 8_2um/manual_segmentation/man_Channel0-T0040.tif')
df_pos8_t45 = pd.DataFrame(measure.regionprops_table(organoid_pos8_t40,properties=['area']))

CVs.loc['organoid_pos2','CV'] = df_pos8_t45['area'].std() / df_pos8_t45['area'].mean()
CVs.loc['organoid_pos2','Category'] = 'Organoid nuclear size'
CVs.loc['organoid_pos2','Counts'] = len(df_pos8_t45)

sb.catplot(CVs,x='Category',y='CV')




