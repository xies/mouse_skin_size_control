#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:31:40 2024

@author: xies
"""

import pandas as pd
import numpy as np
from os import path
import matplotlib.pyplot as plt
import seaborn as sb
import pickle as pkl

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'
df5 = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_features.csv'),index_col=0)
df5['organoidID'] = 5
dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 2_2um/'
df2 = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_features.csv'),index_col=0)
df2['organoidID'] = 2
dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 31_2um/'
df31 = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_features.csv'),index_col=0)
df31['organoidID'] = 31

df = pd.concat((df5,df2,df31),ignore_index=True)
# df = df31
df['organoidID_trackID'] = df['organoidID'].astype(str) + '_' + df['trackID'].astype(str)
regen = df

# Filter out all non-tracked cells
tracks = {trackID:t for trackID,t in df.groupby('organoidID_trackID')}

summary = pd.DataFrame()
# Extract birth, G1S, division
for trackID, track in tracks.items():
    
    # Skip tetraploids
    if trackID == '5_77.0' or trackID == '5_138.0' or trackID == '5_130.0' \
        or trackID == '2_53.0' or trackID == '2_6.0':
        continue
    
    summary.loc[trackID,'organoidID'] = track.iloc[0]['organoidID']
    summary.loc[trackID,'trackID'] = track.iloc[0]['trackID']
    
    # Birth
    I = (track['Phase'] == 'Visible birth')
    summary.loc[trackID,'Birth volume'] = track.iloc[np.where(I)[0][:2]]['Nuclear volume (sm)'].mean()
    
    #first G1S
    I = track['Phase'] == 'G1S'
    summary.loc[trackID,'G1 volume'] = track.iloc[np.where(I)[0][:2]]['Nuclear volume (sm)'].mean()
    # Div
    I = track['Phase'] == 'Visible division'
    summary.loc[trackID,'Division volume'] = track.iloc[np.where(I)[0][:2]]['Nuclear volume (sm)'].mean()
    
    # Find lengths
    summary.loc[trackID,'Birth time'] = track.iloc[0]['Time']
    summary.loc[trackID,'Birth frame'] = track.iloc[0]['Frame']
    I = track['Phase'] == 'G1S'
    if I.sum() > 0:
        summary.loc[trackID,'G1S time'] = track.iloc[np.where(I)[0][0]]['Time'].mean()
    I = track['Phase'] == 'Division'
    if I.sum() > 0:
        summary.loc[trackID,'Division time'] = track.iloc[np.where(I)[0][-1]]['Time'].mean()
    
summary['G1 growth'] = summary['G1 volume'] - summary['Birth volume']
summary['SG2 growth'] = summary['Division volume'] - summary['G1 volume']
summary['Total growth'] = summary['Division volume'] - summary['Birth volume']
summary['G1 duration'] = (summary['G1S time'] - summary['Birth time'])/60 # hours
summary['SG2 duration'] = (summary['Division time'] - summary['G1S time'])/60
summary['Total duration'] = (summary['Division time'] - summary['Birth time'])/60

# sb.lmplot(summary,x='Birth volume',y='G1 growth',hue='organoidID')
sb.regplot(summary,x='Birth volume',y='G1 growth')
plt.gca().set_aspect('equal', 'box')

summary.to_csv(path.join('/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/size_summary.csv'))

#%% Print CV by cell cycle phase

from mathUtils import cvariation_bootstrap, cv_difference_pvalue

CV = pd.DataFrame()

CV.loc['Birth',['CV','LB','UB']] = cvariation_bootstrap(summary['Birth volume'],Nboot=1000,subsample=80)
CV.loc['G1S',['CV','LB','UB']] = cvariation_bootstrap(summary['G1 volume'],Nboot=1000,subsample=80)
CV.loc['Division',['CV','LB','UB']] = cvariation_bootstrap(summary['Division volume'],Nboot=1000,subsample=80)

print(CV)

#%% load old organoid data

dirname = '/Users/xies/Onedrive - Stanford/In vitro/mIOs/Light sheet movies/20200303_194709_09'
homeo = pd.read_csv(path.join(dirname,'size_control.csv'))
# Need to correct for wrong dx
homeo.loc[:,'Birth volume'] *= 1.5**2
homeo.loc[:,'G1 volume'] *= 1.5**2
homeo.loc[:,'G1 growth'] *= 1.5**2

#combine all datasets
summary['Cell type'] = 'Regenerative'

homeo['Cell type'] = 'TA cell'
homeo.loc[homeo['Lgr5'],'Cell type'] = 'Stem cell'

fields2concat = ['Cell type','Birth volume','G1 duration','G1 growth','G1 volume']
df = pd.concat((homeo[fields2concat],summary[fields2concat]))

colors = ['g','m','b'];sb.color_palette(colors)
sb.lmplot(df,x='Birth volume',y='G1 growth',hue='Cell type',palette=colors, robust=False)
plt.gca().set_aspect('equal', 'box')
sb.lmplot(df,x='Birth volume',y='G1 duration',hue='Cell type',palette=colors, robust=False)

sb.catplot(df.reset_index(),y='Birth volume',x='Cell type',kind='violin')
sb.catplot(df.reset_index(),y='G1 volume',x='Cell type',kind='violin')

#%% Print linear regression 

regen = df[df['Cell type'] == 'Regenerative']
stem = df[df['Cell type'] == 'Stem cell']
ta = df[df['Cell type'] == 'TA cell']

import statsmodels.api as sm
_df = stem
X = _df.dropna(subset=['Birth volume','G1 growth'])['Birth volume']
X = sm.add_constant(X)
y = _df.dropna(subset=['Birth volume','G1 growth'])['G1 growth']

model = sm.OLS(y,X).fit()
print(model.conf_int())

#%% Ttests

from scipy.stats import ttest_ind

print('---- Birth sizes-----')
regenerative,stem,ta = [x.dropna(subset='Birth volume') for _,x in df.groupby('Cell type')]

P = ttest_ind(regenerative['Birth volume'],ta['Birth volume'],equal_var=False)
print(f'TA v. regen: {P.pvalue:2f}')

P = ttest_ind(stem['Birth volume'],ta['Birth volume'],equal_var=False)
print(f'TA v. stem: {P.pvalue:2f}')

print('---- G1/S sizes-----')
regenerative,stem,ta = [x.dropna(subset='G1 volume') for _,x in df.groupby('Cell type')]

P = ttest_ind(regenerative['G1 volume'],ta['G1 volume'],equal_var=False)
print(f'TA v. regen: {P.pvalue:2f}')

P = ttest_ind(stem['G1 volume'],ta['G1 volume'],equal_var=False)
print(f'TA v. stem: {P.pvalue:2f}')

#%% Load skin data?

dirname='/Users/xies/OneDrive - Stanford/Skin/Mesa et al/tracked_data_collated'
skin_ts = pd.read_pickle(path.join(dirname,'time_series.pkl'))
skin_summary = pd.DataFrame()

for i, cell in enumerate(skin_ts):
    
    # Birth nuclear volume
    skin_summary.loc[i,'Birth volume'] = cell.iloc[0]['Nucleus']
    
    I = cell.Phase == 'SG2'
    if I.sum() > 0:
        first_sg2_idx = np.where(I)[0][0]
        skin_summary.loc[i,'G1S volume'] = (cell.iloc[first_sg2_idx]['Nucleus'] +cell.iloc[first_sg2_idx-1]['Nucleus'])/2
    
plt.hist(skin_summary['G1S volume'])

# plt.hist(skin_summary['G1S volume'])
# plt.hist(homeo['G1 volume'])
# plt.hist(summary['G1 volume'],weights=1/len(summary))

