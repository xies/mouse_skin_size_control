#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 20:06:53 2023

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from skimage import io, measure
import seaborn as sb
from os import path
from glob import glob
from tqdm import tqdm

import pickle as pkl

from basicUtils import *

dirnames = {}

dirnames['Ablation_R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/F1 black R26 Rbfl DOB 12-27-2022/07-23-2023 R26CreER Rb-fl no tam ablation/R1/'
dirnames['Ablation_R3'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/F1 black R26 Rbfl DOB 12-27-2022/07-26-2023 R25CreER Rb-fl no tam ablation 12h/Black female/R1'
dirnames['Ablation_R5'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/F1 black R26 Rbfl DOB 12-27-2022/07-31-2023 R26CreER Rb-fl no tam ablation 8hr/F1 Black/R1'
dirnames['Ablation_R6'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/F1 black R26 Rbfl DOB 12-27-2022/07-31-2023 R26CreER Rb-fl no tam ablation 8hr/F1 Black/R2'

dirnames['Ablation_R12'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/M5 white R26 RBfl DOB 04-25-2023/08-23-2023 R26CreER Rb-fl no tam ablation 16h/M5 White DOB 4-25-2023/R1/'
dirnames['Ablation_R13'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/M5 white R26 RBfl DOB 04-25-2023/08-23-2023 R26CreER Rb-fl no tam ablation 16h/M5 White DOB 4-25-2023/R2/'
dirnames['Ablation_R14'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/M5 white R26 RBfl DOB 04-25-2023/09-27-2023 R26CreER Rb-fl no tam ablation M5/M5 white DOB 4-25-23/R1'
dirnames['Ablation_R16'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/M5 white R26 RBfl DOB 04-25-2023/10-04-2023 R26CreER Rb-fl no tam ablation M5/M5 white DOB 4-25-23/R1'

dirnames['Ablation_R18'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/M1 M2 K14 Rbfl DOB DOB 06-01-2023/01-13-2024 Ablation K14Cre H2B FUCCI/Black unclipped less leaky DOB 06-30-2023/R2/'
dirnames['Ablation_R20'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Ablation time courses/M1 M2 K14 Rbfl DOB DOB 06-01-2023/01-13-2024 Ablation K14Cre H2B FUCCI/Black right clipped DOB 06-30-2023/R1'

all_tracks = {}
ts_regions = {}
regions = {}
for name,dirname in dirnames.items():
    for mode in ['Ablation','Nonablation']:
        
        # with open(path.join(dirname,'manual_tracking',f'{name}_{mode}_dense.pkl'),'rb') as file:
        #     tracks = pkl.load(file)
        tracks = pd.read_pickle(path.join(dirname,'manual_tracking',f'{name}_{mode}_dense.pkl'))
                    
        all_tracks[name+'_'+mode] = tracks
        
        ts = pd.concat(tracks)
        ts['Distance to ablated cell'] = ts['Distance to ablated cell']*ts.iloc[0].um_per_px
        ts_regions[name+'_'+mode] = ts
        
        df = pd.read_csv(path.join(dirname,f'manual_tracking/{name}_{mode}_dataframe.csv'),index_col=0)
        
        #@todo: move into semiauto so it's pre-computed
        print('--- Save the pre-ablation size as special field ---')
        init_vol = np.ones(len(df))* np.nan
        init_vol_norm = np.ones(len(df)) * np.nan
        for i,t in enumerate(tracks):
            if len(t[t['Frame'] == 0]['Volume'].values) > 0:
                init_vol[i] = t[t['Frame'] == 0]['Volume'].values
                init_vol_norm[i] = t[t['Frame'] == 0]['Volume normal'].values
        df['Initial volume'] = init_vol
        df['Initial volume normal'] = init_vol_norm
        
        regions[name+'_'+mode] = df
        
        # if df.Mouse.iloc[0] == 'WT_Mclip' or df.Mouse.iloc[0] == 'WT_Mnonclip':
        #     df['S phase entry size'] = df['S phase entry size'] / 1.5

df_all = pd.concat(regions,ignore_index=True)
df_all['UniqueID'] = df_all['CellID'].astype(str) + '_' + df_all['Region']
ts_all = pd.concat(ts_regions,ignore_index=True)
ts_all['UniqueID'] = ts_all['CellID'].astype(str) + '_' + ts_all['Region']

df_all.loc[df_all['Mouse'] == 'WT_Mclip','S phase entry size'] = df_all[df_all['Mouse'] == 'WT_Mclip']['S phase entry size']*.7
df_all.loc[df_all['Mouse'] == 'WT_Mnonclip','S phase entry size'] = df_all[df_all['Mouse'] == 'WT_Mnonclip']['S phase entry size']*.7

ablation = ts_all[ts_all['Mode'] == 'Ablation']
nonablation = ts_all[ts_all['Mode'] == 'Nonablation']

#%%

df_all['Mouse_mode'] = df_all['Mouse'] + '_' + df_all['Mode']

# sb.catplot(ts_all,x='Region',hue='Mode',y='Specific GR normal',kind='box')
sb.catplot(df_all,x='Mouse',hue='Mode',y='Exponential growth rate',kind='violin')
sb.stripplot(df_all,x='Mouse',hue='Mode',y='Exponential growth rate',dodge=True)
# sb.catplot(df_all,x='Mouse',hue='Mode',y='Exponential growth rate',kind='box')
# sb.catplot(df_all,x='Mouse',hue='Mode',y='S phase entry size normal',kind='box')

sb.catplot(df_all,x='Mouse',hue='Mode',y='S phase entry size',kind='violin')
sb.stripplot(df_all,x='Mouse',hue='Mode',y='S phase entry size',dodge=True)

plt.ylim([50,200])
# plt.ylim([0,2])

#%%

pairs = zip(range(1),range(1,2))
colors = {'Ablation':'r','Nonablation':'b'}

R = []

for first,second in pairs:
    
    for (_,mode),track in ts_all[ts_all['Region'] == 'Ablation_R18'].groupby(['CellID','Mode']):
        plt.subplot(2,4,first+1)
        plt.plot([0,2],[0,2],'k--')
        
        t1 = track[track['Frame'] == first]['Volume normal'].values
        t2 = track[track['Frame'] == second]['Volume normal'].values
        if len(t1) == 1 and len(t2) == 1:
            R.append([t1,t2])
            plt.scatter(t1,t2,color=colors[mode],alpha=0.2)
        
    # plt.title(f'{first} v. {second}')
    # plt.xlim([0,2]); plt.ylim([0,2])
R = np.squeeze(np.array(R))

#%%

name = 'Ablation_R20'
field2plot = 'Volume'
YMIN = 0
YMAX = 300
tracks = [v for k,v in ts_all[(ts_all['Mode'] == 'Nonablation') & (ts_all['Region'] == name)].groupby(['Region','CellID'])]

plt.subplot(1,2,1)
for t in tracks:
    plt.plot(t.Age, t[field2plot],'b-',alpha=0.3)
plt.xlabel('Time since ablation (h)')
plt.ylabel('Volume (px)')
plt.ylim([YMIN,YMAX])
plt.title('Non neighbors')

tracks = [v for k,v in ts_all[(ts_all['Mode'] == 'Ablation') & (ts_all['Region'] == name)].groupby(['Region','CellID'])]

plt.subplot(1,2,2)
for t in tracks:
    plt.plot(t.Age, t[field2plot],'r-',alpha=0.3)
plt.xlabel('Time since ablation (h)')
plt.ylabel('Volume (px)')
plt.ylim([YMIN,YMAX])
plt.title('Neighbors')

#%% Bin by initial size

initial_sizes = df_all['Initial volume normal']
bins = np.linspace(0,2.5,10)
which_bin = np.digitize(initial_sizes,bins).astype(float)
# Delete the nans
which_bin[ np.isnan(initial_sizes)] = np.nan

for i in range(10):
    I = which_bin == i
    if I.sum() > 5:
        print(f'--Initial size bin = {bins[i]}')
        # T,P = ttest_from_groupby(df_all[I],'Mode','Exponential growth rate')
        print(df_all[I].groupby(['Mouse','Mode'])['S phase entry size normal'].mean())
        # print(df_all[I].groupby(['Mouse','Mode'])['Exponential growth rate'].mean())
        

#%%

from basicUtils import plot_bin_means

# ts_all['Specific GR'] = ts_all['Growth rate'] / ts_all['Volume']
# ts_all['Specific GR normal'] = ts_all['Growth rate normal'] / ts_all['Volume normal']
# sb.catplot(ts_all,x='Mode',y='Specific GR normal',kind='box')
# sb.catplot(df_all,hue='Mode',y='Exponential growth rate',x='Region',kind='box')

# sb.catplot(ts_all,x='Mode',y='Specific GR normal',kind='box',hue='Region')

sb.lmplot(df_all,x='Distance to ablated cell',y='Exponential growth rate',
          fit_reg=False,scatter_kws={'alpha':.1}, hue='Mode')
plot_bin_means(df_all['Distance to ablated cell'],df_all['Exponential growth rate'],
               bin_edges=10,color='red',bin_style='percentile')

# plt.figure()
# sb.lmplot(ts_all,x='Distance to ablated cell',y='Specific GR normal', scatter_kws={'alpha':.1})

D = pd.DataFrame(ts_all.groupby(['CellID','Region','Mode'])['Distance to ablated cell'].mean())
D = D.reset_index()
sb.histplot(D,x='Distance to ablated cell',hue='Mode',bins=50,element='poly')
df_all = pd.merge(df_all,D,on=['Region','Mode','CellID'])

#%%

from basicUtils import ttest_from_groupby

for mousename,mouse in df_all.groupby('Mouse'):
    
    print(f'--- {mousename} ---')
    # T,P = ttest_from_groupby(mouse,'Mode','S phase entry size')
    # print(f'S phase entry size, P = {P}')
    
    T,P = ttest_from_groupby(mouse,'Mode','S phase entry size normal')
    print(f'S phase entry size normal, P = {P}')
    
    T,P = ttest_from_groupby(mouse,'Mode','Exponential growth rate')
    print(f'Exponential growth rate, P = {P}')


# for mousename,mouse in ts_all.groupby('Mouse'):
#     print(f'--- {mousename} ---')
#     T,P = ttest_from_groupby(mouse,'Mode','Specific GR')
#     print(f'Specific GR, P = {P}')
    
#     T,P = ttest_from_groupby(mouse,'Mode','Specific GR normal')
#     print(f'Specific GR nromal, P = {P}')





