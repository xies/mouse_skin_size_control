#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:56:05 2024

@author: xies
"""

import numpy as np
import pandas as pd
from os import path
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sb
from basicUtils import nonans,plot_bin_means
from mathUtils import cvariation_ci_bootstrap

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/Shuyuan HMECs HDBD/'

filelist = ['HMECs_CDKhighlow.xlsx']

time  = pd.read_excel(path.join(dirname,filelist[0]),sheet_name='Frame_align_G1S').drop(columns=['Unnamed: 0.5','Unnamed: 0.1','Unnamed: 0','Unnamed: 0.2','Unnamed: 0.3','Unnamed: 0.4','Unnamed: 0.1.1'])
# hdhb  = pd.read_excel(path.join(dirname,filelist[0]),sheet_name='HDHB_ratio').drop(columns=['Unnamed: 0.5','Unnamed: 0.1','Unnamed: 0','Unnamed: 0.2','Unnamed: 0.3','Unnamed: 0.4','Unnamed: 0.1.1'])
nuc_vol  = pd.read_excel(path.join(dirname,filelist[0]),sheet_name='Nucleus volume').drop(columns=['Unnamed: 0.5','Unnamed: 0.1','Unnamed: 0','Unnamed: 0.2','Unnamed: 0.3','Unnamed: 0.4','Unnamed: 0.1.1'])

#%%

Nframe,Ncells = time.shape
# determine g1 length for each cell

df = pd.DataFrame(index=time.columns)

for cell in time.columns:

    df.loc[cell,'G1 length'] = -np.nanmin(time[cell])
    
    Ig1s = np.where(time[cell] == 0)[0][0]
    df.loc[cell,'G1 size'] = (nuc_vol[cell].iloc[Ig1s-1:Ig1s+2].values).mean()
    
    df.loc[cell,'Birth size'] = nuc_vol[cell].iloc[3:6].mean()
    
df['G1 growth'] = df['G1 size'] - df['Birth size']
df['CDK high'] = df['G1 length'] < 6*3

#%%

(_,cdk_low),(_,cdk_high) = df.groupby('CDK high')

sb.lmplot(df,x = 'Birth size',y='G1 growth',hue='CDK high')

plot_bin_means(cdk_low['Birth size'],cdk_low['G1 growth'],bin_edges=10,color='r')
plot_bin_means(cdk_high['Birth size'],cdk_high['G1 growth'],bin_edges=10,color='r')

p = np.polyfit(cdk_low['Birth size'],cdk_low['G1 growth'],1)
print(f'Low {p[0]}')
p = np.polyfit(cdk_high['Birth size'],cdk_high['G1 growth'],1)
print(f'High {p[0]}')

#%% Look at CV -- pan cell cycle

g1_size_high = nuc_vol[time < 0].loc[:,df['CDK high']]
g1_size_low = nuc_vol[time < 0].loc[:,~df['CDK high']]

CV_g1_high = cvariation_ci_bootstrap(nonans(g1_size_high), 1000)[0]
CV_g1_low = cvariation_ci_bootstrap(nonans(g1_size_low), 1000)[0]
print(f'{CV_g1_high}')
print(f'{CV_g1_low}')

sg2_size_high = nuc_vol[time >= 0].loc[:,df['CDK high']]
sg2_size_low = nuc_vol[time >= 0].loc[:,~df['CDK high']]

CV_sg2_high = cvariation_ci_bootstrap(nonans(sg2_size_high), 1000)[0]
CV_sg2_low = cvariation_ci_bootstrap(nonans(sg2_size_low), 1000)[0]

print(f'{CV_sg2_high}')
print(f'{CV_sg2_low}')

#%% Look at CV -- pan cell cycle


