#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 17:38:58 2021

@author: xies
"""

import numpy as np
import pandas as pd
from os import path
import pickle as pkl
import seaborn as sb

#%%

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/05-03-2021 Rb-fl/M2 RB-KO/R1/'

with open(path.join(dirname,'final_trackseg','complete_cycles_final.pkl'),'rb') as file:
    tracks = pkl.load(file)    
df = pd.read_csv(path.join(dirname,'final_trackseg','cell_dataframe.pkl'))
df['Dataset'] = '05-03-2021'
df['Genotype'] = 'RB-KO'

tracks_rbko = tracks
df_rbko = df

for track in tracks:
    track['Genotype'] = 'RB-KO'


dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/05-03-2021 Rb-fl/M1 WT/R1/'
with open(path.join(dirname,'final_trackseg','complete_cycles_final.pkl'),'rb') as file:
    tracks = pkl.load(file)    
df = pd.read_csv(path.join(dirname,'final_trackseg','cell_dataframe.pkl'))
df['Dataset'] = '05-03-2021'
df['Genotype'] = 'WT (sibling)'

tracks_wt = tracks
df_wt = df

for track in tracks:
    track['Genotype'] = 'WT sibling'

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/10-20-2021/WT/R1/'

with open(path.join(dirname,'final_trackseg','complete_cycles_final.pkl'),'rb') as file:
    tracks = pkl.load(file)    
df = pd.read_csv(path.join(dirname,'final_trackseg','cell_dataframe.pkl'))
df['Dataset'] = '10-20-2021'
df['Genotype'] = 'WT (non-sibling)'

for track in tracks:
    track['Genotype'] = 'WT non-sibling'

tracks_wt2 = tracks
df_wt2 = df
# df_wt = pd.concat((df_wt,df),ignore_index=True)

# Put all data together
ts = pd.concat((pd.concat(tracks_rbko),
                pd.concat(tracks_wt,),
                pd.concat(tracks_wt2)),ignore_index=True)

df = pd.concat((df_rbko,df_wt,df_wt2),ignore_index=True)
# # Load final tracks
# with open(path.join(dirname,'manual_tracking','complete_cycles_fixed.pkl'),'rb') as file:
#     tracks = pkl.load(file)
    
# with open('/Users/xies/Box/Mouse/Skin/Mesa et al/Pickled/cell_summary.pkl','rb') as file:
#     wt_cells = pkl.load(file, encoding='latin1')

# with open('/Users/xies/Box/Mouse/Skin/Mesa et al/Pickled/time_series.pkl','rb') as file:
#     wt_ts = pkl.load(file, encoding='latin1')#

#%% Plot average size

sb.catplot(data = ts, x = 'Genotype', y = 'Volume',kind='violin')
plt.ylabel('Nuclear volume averaged over whole cell cycle')
print('Size over whole cycle')
print(ts.groupby('Genotype').mean()['Volume'] * 0.2920097**2)

ts_g1 = ts[ts['Phase'] == 'G1']
ts_sg2 = ts[ts['Phase'] == 'SG2']

sb.catplot(data = ts_g1, x = 'Genotype', y = 'Volume',kind='violin')
plt.ylabel('Nuclear volume averaged over G1')
print('Size over G1')
print(ts_g1.groupby('Genotype').mean()['Volume'] * 0.2920097**2)

sb.catplot(data = ts_sg2, x = 'Genotype', y = 'Volume',kind='violin')
plt.ylabel('Nuclear volume averaged over SG2')
print('Size over SG2')
print(ts_sg2.groupby('Genotype').mean()['Volume'] * 0.2920097**2)


#%% Plot cell cycle times
plt.figure()
sb.histplot(data=df,x='Cycle length',hue='Genotype',bins=7, multiple='layer',element='step',
            common_bins = False,fill=True)
plt.xlabel('Cell cycle duration (h)')


plt.figure()
sb.histplot(data=df,x='G1 length',hue='Genotype',bins=7, multiple='layer',element='step',
            common_bins = False,fill=True)
plt.xlabel('G1 duration (h)')


plt.figure()
sb.histplot(data=df,x='SG2 length',hue='Genotype',bins=7, multiple='layer',element='step',
            common_bins = False,fill=True)
plt.xlabel('SG2 duration (h)')

print(df.groupby('Genotype').mean()['G1 length'])
print(df.groupby('Genotype').mean()['SG2 length'])
print(df.groupby('Genotype').mean()['Cycle length'])

#%%  Plot size control (time)

# plt.figure()
# sb.regplot(data = df_rbko, x='Birth size', y = 'Cycle length', y_jitter=True)
# sb.regplot(data = df_wt, x='Birth size', y = 'Cycle length', y_jitter=True)
# sb.regplot(data = df_wt2, x='Birth size', y = 'Cycle length', y_jitter=True)
sb.lmplot(data= df, x='Birth size',y='Cycle length',y_jitter=True, hue='Genotype')
plt.xlim([0,500]); plt.ylim([0,150])
# plt.legend(['RB-KO','WT (sibling)','WT (non-sib)'])

sb.lmplot(data= df, x='Birth size',y='G1 length',y_jitter=True, hue='Genotype')
plt.xlim([0,500]); plt.ylim([0,120])
# plt.figure()
# sb.regplot(data = df_rbko, x='Birth size', y = 'G1 length', y_jitter=True) 
# sb.regplot(data = df_wt, x='Birth size', y = 'G1 length', y_jitter=True) 
# sb.regplot(data = df_wt2, x='Birth size', y = 'G1 length', y_jitter=True) 
# plt.legend(['RB-KO','WT (sibling)','WT (non-sib)'])

sb.lmplot(data= df, x='Birth size',y='SG2 length',y_jitter=True, hue='Genotype')
plt.xlim([0,500]);

#%%  Plot size control (growth)

def nonans(x,y):
    I = ~np.isnan(x)
    I = I & ~np.isnan(y)
    return x[I],y[I]

def pearson_r(x,y):
    x,y = nonans(x,y)
    return np.corrcoef(x,y)[0,1]

def slope(x,y):
    x,y = nonans(x,y)
    p = np.polyfit(x,y,1)
    return p[0]
    
# G1 exit
plt.figure()
sb.regplot(data = df_rbko, x='Birth size', y = 'G1 growth')
sb.regplot(data = df_wt, x='Birth size', y = 'G1 growth')
sb.regplot(data = df_wt2, x='Birth size', y = 'G1 growth')
plt.xlim([0,500])
plt.ylim([0,500])
plt.legend(['RB-KO','WT (sibling)','WT (non-sib)'])

print('\n\n-----RB-KO: G1 growth------')
R = pearson_r(df_rbko['Birth size'],df_rbko['G1 growth'])
print(f'G1 Pearson R = {R}')

p = slope(df_rbko['Birth size'],df_rbko['G1 growth'])
print(f'G1 Slope m = {p}')

print('\n\n-----WT sib: G1 growth------')
R = pearson_r(df_wt['Birth size'],df_wt['G1 growth'])
print(f'G1 Pearson R = {R}')

p = slope(df_wt['Birth size'],df_wt['G1 growth'])
print(f'G1 Slope m = {p}')


print('\n\n-----WT nonsib: G1 growth------')
R = pearson_r(df_wt2['Birth size'],df_wt2['G1 growth'])
print(f'G1 Pearson R = {R}')

p = slope(df_wt2['Birth size'],df_wt2['G1 growth'])
print(f'G1 Slope m = {p}')


# SG2 growth
print('-------')
plt.figure()
sb.regplot(data = df_rbko, x='Birth size', y = 'SG2 growth')
sb.regplot(data = df_wt, x='Birth size', y = 'SG2 growth')
sb.regplot(data = df_wt2, x='Birth size', y = 'SG2 growth')
plt.xlim([0,500])
plt.ylim([0,500])
plt.legend(['RB-KO','WT (sibling)','WT (non-sib)'])

print('\n\n-----RB-KO: SG2 growth------')
R = pearson_r(df_rbko['Birth size'],df_rbko['SG2 growth'])
print(f'G1 Pearson R = {R}')

p = slope(df_rbko['Birth size'],df_rbko['SG2 growth'])
print(f'SG2 Slope m = {p}')

print('\n\n-----WT sib: SG2 growth------')
R = pearson_r(df_wt['Birth size'],df_wt['SG2 growth'])
print(f'SG2 Pearson R = {R}')

p = slope(df_wt['Birth size'],df_wt['SG2 growth'])
print(f'SG2 Slope m = {p}')


print('\n\n-----WT nonsib: SG2 growth------')
R = pearson_r(df_wt2['Birth size'],df_wt2['SG2 growth'])
print(f'SG2 Pearson R = {R}')

p = slope(df_wt2['Birth size'],df_wt2['SG2 growth'])
print(f'SG2 Slope m = {p}')


# Whole cycle
print('-------')
plt.figure()
sb.regplot(data = df_rbko, x='Birth size', y = 'Total growth')
sb.regplot(data = df_wt, x='Birth size', y = 'Total growth')
sb.regplot(data = df_wt2, x='Birth size', y = 'Total growth')
plt.xlim([-100,500])
plt.ylim([-100,500])
plt.legend(['RB-KO','WT (sibling)','WT (non-sib)'])

print('\n\n-----RB-KO: Total growth------')
R = pearson_r(df_rbko['Birth size'],df_rbko['Total growth'])
print(f'Total Pearson R = {R}')

p = slope(df_rbko['Birth size'],df_rbko['Total growth'])
print(f'Total Slope m = {p}')



print('\n\n-----WT sib: Total growth ------')
R = pearson_r(df_wt['Birth size'],df_wt['Total growth'])
print(f'Total Pearson R = {R}')

p = slope(df_wt['Birth size'],df_wt['Total growth'])
print(f'Total Slope m = {p}')


print('\n\n-----WT nonsib: Total growth ------')
R = pearson_r(df_wt2['Birth size'],df_wt2['Total growth'])
print(f'Total Pearson R = {R}')

p = slope(df_wt2['Birth size'],df_wt2['Total growth'])
print(f'Total Slope m = {p}')

  