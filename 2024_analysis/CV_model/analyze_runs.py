#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:42:05 2024

@author: xies
"""

import pandas as pd
import numpy as np
from numpy import random
import seaborn as sb
import matplotlib.pylab as plt

from os import path
from glob import glob
import pickle as pkl
import copy
from tqdm import tqdm
from scipy import stats
from basicUtils import nonan_pairs, nonans
from mathUtils import cvariation_bootstrap


def plot_growth_curves_population(pop):
    
    plt.figure()
    for cell in pop.values():
        ts = cell.ts.dropna()
        t = ts['Time']
        v = ts['Volume']
        p = ts['Phase']
        
        t_g1 = t[p =='G1']
        v_g1 = v[p =='G1']
        plt.plot(t_g1,v_g1,'b-',alpha=0.1)
        
        t_g2 = t[p =='S/G2/M']
        v_g2 = v[p =='S/G2/M']
        plt.plot(t_g2,v_g2,'r-',alpha=0.1)
        plt.xlabel('Time (h)')
        plt.ylabel('Cell volume (fL)')
        plt.legend(['G1','S/G2/M'])
        
def extract_CVs(population,measurement_field='Measured volume'):

    collated = []
    for key,cell in pop2analyze.items():
        ts = cell.ts.dropna().copy()
        btime = ts.iloc[0]['Time']
        ts['Age'] = ts['Time'] - btime
        collated.append(ts)
        
    collated = pd.concat(collated,ignore_index=True)
    CV = pd.DataFrame()
    for phase,x in collated.groupby('Phase')[measurement_field]:
        CV.loc['Time',phase] = x.std()/x.mean()
    
    time = np.vstack( [ cell.ts['Time'].astype(float) for cell in pop2analyze.values() ])
    size = np.vstack( [ cell.ts[measurement_field].astype(float) for cell in pop2analyze.values() ])
    phases = np.vstack( [ cell.ts['Phase'] for cell in pop2analyze.values() ])
    
    # for t in range(max_iter):
    # Only look at last time point
    t = max_iter-1
    p = phases[:,t]
    s = size[p == 'G1',t]
    if (len(s)>3):
        CV_time_g1 = s.std()/s.mean()
    s = size[p == 'S/G2/M',t]
    if (len(s)>3):
        CV_time_sg2 = s.std()/s.mean()
    
    CV.loc['Population','G1'] = CV_time_g1
    CV.loc['Population','S/G2/M'] = CV_time_sg2
    
    return CV

def extract_time_window_after_g1s(pop2analyze,num_frames_to_extract,
                                  field2extract='Measured volume'):
    Ncells = len(pop2analyze)
    windowed = np.ones((Ncells,num_frames_to_extract)) * np.nan
    for i,(_,cell) in enumerate(pop2analyze.items()):
        if not np.isnan(cell.g1s_frame):
            windowed[i,:] = cell.ts[field2extract][cell.g1s_frame:cell.g1s_frame+num_frames_to_extract]
    return windowed



#%% Load runs and collate stats for each model parameter set

# dirname = '/Users/xies/OneDrive - Stanford/In vitro/CV from snapshot/CV model/G1timer_SG2sizer_asym05_grfluct05/'
# params = pd.read_csv(path.join(dirname,'params.csv'),index_col=0)

# with open(path.join(dirname,'adders.pkl'),'rb') as f:
#     runs = pkl.load(f)

CVs = {}
for model_name,pop2analyze in tqdm(runs.items()):
    CVs[model_name] = extract_CVs(pop2analyze,measurement_field='Measured volume')

df = pd.DataFrame(index=CVs.keys(),columns=['G1 model','SG2M model'
                                            ,'G1 CV','SG2M CV','CVdiff','Birth CV','G1S CV','Div CV'
                                            ,'G1 size control slope','SG2M size control slope'
                                            ,'Birth size','G1 size','Div size'
                                            ,'G1 growth','SG2M growth','Total growth','G1/G2 growth ratio'
                                            ,'Cycle duration','G1 duration','SG2M duration'
                                            ])

birth_CV_by_gen = {}
div_CV_by_gen = {}
for model_name,_df in CVs.items():
    
    df.loc[model_name,'G1 model'] = params.loc[model_name,'G1S_model']
    df.loc[model_name,'SG2M model'] = params.loc[model_name,'SG2M_model']
    df.loc[model_name,'G1 CV'] =  _df.loc['Population','G1']
    df.loc[model_name,'SG2M CV'] =  _df.loc['Population','S/G2/M']
    df.loc[model_name,'CVdiff'] = _df.loc['Population','G1']- _df.loc['Population','S/G2/M']
    df.loc[model_name,'CVratio'] = _df.loc['Population','G1']- _df.loc['Population','S/G2/M']

    # Size
    bsize = np.array([c.birth_size for c in runs[model_name].values()])
    g1size = np.array([c.g1s_size for c in runs[model_name].values()])
    dsize = np.array([c.div_size for c in runs[model_name].values()])
    g1_growth = g1size - bsize
    sg2_growth = dsize - g1size
    total_growth = dsize - bsize
    
    df.loc[model_name,'Birth size'] = np.nanmean(bsize)
    df.loc[model_name,'G1 size'] = np.nanmean(g1size)
    df.loc[model_name,'Div size'] = np.nanmean(dsize)
    df.loc[model_name,'G1 growth'] = np.nanmean(g1_growth)
    df.loc[model_name,'SG2M size'] = np.nanmean(sg2_growth)
    df.loc[model_name,'Total growth'] = np.nanmean(total_growth)
    df.loc[model_name,'G1 growth ratio'] = np.nanmean(g1_growth/total_growth)
    
    # CV of size
    df.loc[model_name,['[Birth CV','Birth CV LB','Birth CV UB']] = \
        cvariation_bootstrap(bsize,Nboot=1000,subsample=100)
    df.loc[model_name,['G1S CV','G1S CV LB','G1S CV UB']] = \
        cvariation_bootstrap(g1size,Nboot=1000,subsample=100)
    df.loc[model_name,['Div CV','Div CV LB','Div CV UB']] = \
        cvariation_bootstrap(dsize,Nboot=1000,subsample=100)
    _df = pd.DataFrame()
    _df['Generation'] = [c.generation for c in runs[model_name].values()]
    _df['Birth size'] = bsize
    _df['G1 size'] = g1size
    _df['Div size'] = dsize
    birth_CV_by_gen[model_name] = _df.groupby('Generation')['Birth size'].apply(stats.variation)
    div_CV_by_gen[model_name] = _df.dropna(subset='Div size').groupby('Generation')['Div size'].apply(stats.variation)
    l = birth_CV_by_gen[model_name]
    df.loc[model_name,'bCV by gen'] = np.polyfit(range(len(l)),l,1)[0]

    # Time
    g1 = np.array([cell.g1s_time - cell.ts['Time'].min() for cell in runs[model_name].values()])
    div = np.array([cell.div_time - cell.ts['Time'].min() for cell in runs[model_name].values()])
    
    df.loc[model_name,'Cycle duration'] = np.nanmean(div)
    df.loc[model_name,'G1 duration'] = np.nanmean(g1)
    df.loc[model_name,'G1 duration std'] = np.nanstd(g1)
    df.loc[model_name,'SG2M duration'] = np.nanmean(div - g1)
    
    X,Y = nonan_pairs(bsize,g1_growth)
    p = np.polyfit(X,Y,1)
    df.loc[model_name,'G1 size control slope'] = p[0]
    X,Y = nonan_pairs(g1size,sg2_growth)
    p = np.polyfit(X,Y,1)
    df.loc[model_name,'SG2M size control slope'] = p[0]
    X,Y = nonan_pairs(g1size,total_growth)
    p = np.polyfit(X,Y,1)
    df.loc[model_name,'Final size control slope'] = p[0]

df.to_csv(path.join(dirname,'model_summary.csv'))

#%% Collate 'birth-aligned', 'G1S aligned' and 'normalized' time series

pop2analyze = runs['timer40_sizer']

common_time = np.linspace(0,1,20)

birth_aligned = {}
g1s_aligned = {}
normalized = {}
for key,cell in pop2analyze.items():
    
    ts = cell.ts
    I = ~np.isnan(cell.ts['Time'].astype(float))
    t = cell.ts['Time'][I] - cell.ts['Time'][I].iloc[0]
    v = cell.ts['Measured volume'][I]
    birth_aligned[key] = (t,v)
    
    if np.any(cell.ts['Phase'] == 'S/G2/M'):
        v = cell.ts['Measured volume'][I]
        t = cell.ts['Time'][I] - cell.ts['Time'][I].iloc[np.where(cell.ts['Phase'][I] == 'S/G2/M')[0][0]]
        g1s_aligned[key] = (t,v)
    
        if cell.divided:
            I_g1 = t < 0
            v_g1 = v[I_g1].astype(float)
            t_g1 = np.linspace(0,1,len(v_g1))
            vg1_interp = np.interp(common_time,t_g1,v_g1)
            
            I_g1 = t < 0
            v_g1 = v[I_g1].astype(float)
            t_g1 = np.linspace(0,1,len(v_g1))
            vg1_interp = np.interp(common_time,t_g1,v_g1)
            
            I_g2 = t >= 0
            v_g2 = v[I_g2].astype(float)
            t_g2 = np.linspace(0,1,len(v_g2))
            vg2_interp = np.interp(common_time,t_g2,v_g2)
            
            t_interp = np.linspace(0,1,100)
            v_interp = np.hstack((vg1_interp,vg2_interp))
        
            normalized[key] = (t_interp,v_interp)

    
# Construct birth-aligned matrix
num_frames = np.array([len(v) for key,(t,v) in birth_aligned.items()])
Ncells = len(birth_aligned)
birth_ts = np.ones((Ncells,num_frames.max())) * np.nan
for i,(_,(_,v)) in enumerate(birth_aligned.items()):
    birth_ts[i,0:len(v)] = v.values

birth_aligned_CV = np.ones(num_frames.max()) * np.nan
for t in range(num_frames.max()):
    v = nonans(birth_ts[:,t])
    if len(v) > 200:
        birth_aligned_CV[t] = cvariation_bootstrap(v,Nboot = 200)[0]
plt.plot(birth_aligned_CV);
plt.xlabel('Time since birth (h)')
plt.ylabel('size CV')

# G1s aligned matrix
num_g1_frames = np.array([len(t[t<0]) for key,(t,v) in g1s_aligned.items()])
num_sg2_frames = np.array([len(t[t>=0]) for key,(t,v) in g1s_aligned.items()])
g1s_transition_frame = num_g1_frames.max()
Ncells = len(birth_aligned)
g1s_ts = np.ones((Ncells,num_g1_frames.max() + num_sg2_frames.max())) * np.nan
for i,(_,(t,v)) in enumerate(g1s_aligned.items()):
    v_g1 = v[t<0]
    g1s_ts[i,g1s_transition_frame-len(v_g1):g1s_transition_frame] = v_g1.values
    v_g2 = v[t>=0]
    g1s_ts[i,g1s_transition_frame:g1s_transition_frame+len(v_g2)] = v_g2.values
    

g1s_aligned_CV = np.ones(g1s_ts.shape[1]) * np.nan
for t in range(g1s_ts.shape[1]):
    v = nonans(g1s_ts[:,t])
    if len(v) > 200:
        g1s_aligned_CV[t] = cvariation_bootstrap(v,Nboot = 200)[0]
plt.plot(g1s_aligned_CV); plt.vlines(x=g1s_transition_frame,ymin=0,ymax=0.3,color='r')
plt.xlabel('Time since G1/S (h)')
plt.ylabel('size CV')


# Normalized matrix
norm_ts = np.array([v for _,(t,v) in normalized.items()])

norm_CV = np.nanstd(norm_ts,axis=0) / np.nanmean(norm_ts,axis=0)
plt.plot(norm_CV); plt.vlines(x=20,ymin=0,ymax=0.3,color='r')
plt.xlabel('Normalized cell cycle time (a.u.)')
plt.ylabel('size CV')

#%%

df['Model name'] = df.index
sb.scatterplot(df,x='CVdiff',y='bCV by gen',hue='G1 model');plt.ylabel('CV increase per generation');plt.xlabel('CV_g1  - CV_sg2m')

plt.figure()
sb.scatterplot(df,x='G1 CV',y='SG2M CV',hue='G1 model')
plt.scatter(emp_g1_cv,emp_sg2_cv,color='r')

# plt.figure()
# plt.subplot(2,1,1)
# sb.scatterplot(df,x='G1 growth ratio',y='G1 CV',hue='G1 model')
# plt.subplot(2,1,2)
# sb.scatterplot(df,x='G1 growth ratio',y='SG2M CV',hue='G1 model')

plt.figure()
sb.scatterplot(df,x='G1 size control slope',y='CVdiff',hue='G1 model')

plt.figure()
sb.scatterplot(df,x='G1 duration',y='G1S CV',hue='G1 model')
plt.figure()
sb.scatterplot(df,x='G1 duration',y='Birth CV',hue='G1 model')
df['BCV-G1SCV'] = df['Birth CV'] - df['G1S CV']
plt.figure()
sb.scatterplot(df,x='G1 duration',y='BCV-G1SCV',hue='G1 model')

df['Diff between G1S CV and total G1 CV'] = df['G1S CV'] - df['G1 CV']
sb.scatterplot(df,x='Model name',y='Diff between G1S CV and total G1 CV',hue='G1 model')
plt.gca().tick_params(axis='x', labelrotation=45)
plt.tight_layout()

# plot_growth_curves_population(runs['timer30_sizer'])
# 


