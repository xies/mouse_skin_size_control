#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:15:45 2024

@author: xies
"""


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from FlowCytometryTools import FCMeasurement
from glob import glob
from os import path

from SelectFromCollection import SelectFromCollection
from matplotlib.path import Path

from mathUtils import cvariation_ci, cvariation_ci_bootstrap

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/Flow FUCCI/MDCK time course'
all_singlets = []

#%% Day 1

filelist = glob(path.join(dirname,'*Group_Day1.fcs'))
df = FCMeasurement(ID='Test Sample', datafile=filelist[0])

# Gate on singlets
pts = plt.scatter(df['VL1-H'],df['VL1-W'],alpha=0.005)
plt.xlabel('DAPI-height');plt.ylabel('DAPI-width');
selector = SelectFromCollection(plt.gca(), pts)

#%%

verts = np.array(selector.poly.verts)
x = verts[:,0];y = verts[:,1]
p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(df['VL1-H'],df['VL1-W'])])

singlets = df[I]
singlets['Day'] = 1

#%% Gate cell cycle based on DAPI only

# Set Cdt threshold
th = 0.51e6
plt.hist(singlets['VL1-A'],100);plt.xlabel('VL1-A')
plt.vlines(x=th,ymin=0,ymax=1000,color='r')
singlets['High_DAPI'] = True
singlets.loc[singlets['VL1-A'] < th,'High_DAPI'] = False

all_singlets.append(singlets)

#%% Day 2

filelist = glob(path.join(dirname,'*Group_Day2.fcs'))
df = FCMeasurement(ID='Test Sample', datafile=filelist[0])

# Gate on singlets
pts = plt.scatter(df['VL1-H'],df['VL1-W'],alpha=0.005)
plt.xlabel('DAPI-height');plt.ylabel('DAPI-width');
selector = SelectFromCollection(plt.gca(), pts)

#%%

verts = np.array(selector.poly.verts)
x = verts[:,0];y = verts[:,1]
p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(df['VL1-H'],df['VL1-W'])])

singlets = df[I]
singlets['Day'] = 2

#%% Gate cell cycle based on DAPI only

# Set Cdt threshold
th = 0.47e6
plt.hist(singlets['VL1-A'],100);plt.xlabel('VL1-A')
plt.vlines(x=th,ymin=0,ymax=1000,color='r')
singlets['High_DAPI'] = True
singlets.loc[singlets['VL1-A'] < th,'High_DAPI'] = False

#%%

all_singlets.append(singlets)

#%% Day 3

filelist = glob(path.join(dirname,'*Group_Day3.fcs'))
df = FCMeasurement(ID='Test Sample', datafile=filelist[0])

# Gate on singlets
pts = plt.scatter(df['VL1-H'],df['VL1-W'],alpha=0.005)
plt.xlabel('DAPI-height');plt.ylabel('DAPI-width');
selector = SelectFromCollection(plt.gca(), pts)

#%%

verts = np.array(selector.poly.verts)
x = verts[:,0];y = verts[:,1]
p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(df['VL1-H'],df['VL1-W'])])

singlets = df[I]
singlets['Day'] = 3

#%% Gate cell cycle based on DAPI only

# Set Cdt threshold
th = 0.45e6
plt.hist(singlets['VL1-A'],100);plt.xlabel('VL1-A')
plt.vlines(x=th,ymin=0,ymax=4000,color='r')
singlets['High_DAPI'] = True
singlets.loc[singlets['VL1-A'] < th,'High_DAPI'] = False

#%%

all_singlets.append(singlets)

#%% Day 4

filelist = glob(path.join(dirname,'*Group_Day4.fcs'))
df = FCMeasurement(ID='Test Sample', datafile=filelist[0])

# Gate on singlets
pts = plt.scatter(df['VL1-H'],df['VL1-W'],alpha=0.005)
plt.xlabel('DAPI-height');plt.ylabel('DAPI-width');
selector = SelectFromCollection(plt.gca(), pts)

#%%

verts = np.array(selector.poly.verts)
x = verts[:,0];y = verts[:,1]
p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(df['VL1-H'],df['VL1-W'])])

singlets = df[I]
singlets['Day'] = 4

#%% Gate cell cycle based on DAPI only

# Set Cdt threshold
th = 0.35e6
plt.hist(singlets['VL1-A'],100);plt.xlabel('VL1-A')
plt.vlines(x=th,ymin=0,ymax=4000,color='r')
singlets['High_DAPI'] = True
singlets.loc[singlets['VL1-A'] < th,'High_DAPI'] = False

#%%

all_singlets.append(singlets)

#%% Day 5

filelist = glob(path.join(dirname,'*Group_Day5.fcs'))
df = FCMeasurement(ID='Test Sample', datafile=filelist[0])

# Gate on singlets
pts = plt.scatter(df['VL1-H'],df['VL1-W'],alpha=0.005)
plt.xlabel('DAPI-height');plt.ylabel('DAPI-width');
selector = SelectFromCollection(plt.gca(), pts)

#%%

verts = np.array(selector.poly.verts)
x = verts[:,0];y = verts[:,1]
p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(df['VL1-H'],df['VL1-W'])])

singlets = df[I]
singlets['Day'] = 5

#%% Gate cell cycle based on DAPI only

# Set Cdt threshold
th = 0.3e6
plt.hist(singlets['VL1-A'],100);plt.xlabel('VL1-A')
plt.vlines(x=th,ymin=0,ymax=4000,color='r')
singlets['High_DAPI'] = True
singlets.loc[singlets['VL1-A'] < th,'High_DAPI'] = False

#%%

all_singlets.append(singlets)

#%%

df = pd.concat(all_singlets)
df.to_csv(path.join(dirname,'singlets.csv'))

#%%

CV_FSC = df.groupby(['Day','High_DAPI'])['FSC-A'].std()/ df.groupby(['Day','High_DAPI'])['FSC-A'].mean()
CV_FSC = CV_FSC.reset_index()
CV_FSC = CV_FSC.rename(columns={'FSC-A':'CV of FSC'})
CV_SSC = df.groupby(['Day','High_DAPI'])['SSC-A'].std()/ df.groupby(['Day','High_DAPI'])['SSC-A'].mean()
CV_SSC = CV_SSC.reset_index()
CV_SSC = CV_SSC.rename(columns={'SSC-A':'CV of SSC'})
CVs = CV_FSC.merge(CV_SSC,on=['Day','High_DAPI'])

#%%

Nboot = 1000

for day,_df in df.groupby('Day'):
    
    (_,twoN),(_,fourN) = _df.groupby('High_DAPI')
    
    lb_2n,ub_2n = cvariation_ci_bootstrap(twoN['FSC-A'],Nboot)
    lb_4n,ub_4n = cvariation_ci_bootstrap(fourN['FSC-A'],Nboot)
    
    CVs.loc[(CVs['Day']==day) & (~CVs['High_DAPI']),'CV of FSC 5%'] = lb_2n
    CVs.loc[(CVs['Day']==day) & (~CVs['High_DAPI']),'CV of FSC 95%'] = ub_2n
    CVs.loc[(CVs['Day']==day) & (~CVs['High_DAPI']),'CV of FSC 5%'] = lb_2n
    CVs.loc[(CVs['Day']==day) & (~CVs['High_DAPI']),'CV of FSC 95%'] = ub_2n
    CVs.loc[(CVs['Day']==day) & (CVs['High_DAPI']),'CV of FSC 5%'] = lb_4n
    CVs.loc[(CVs['Day']==day) & (CVs['High_DAPI']),'CV of FSC 95%'] = ub_4n
    
    lb_2n,ub_2n = cvariation_ci_bootstrap(twoN['SSC-A'],Nboot)
    lb_4n,ub_4n = cvariation_ci_bootstrap(fourN['SSC-A'],Nboot)
    
    CVs.loc[(CVs['Day']==day) & (~CVs['High_DAPI']),'CV of SSC 5%'] = lb_2n
    CVs.loc[(CVs['Day']==day) & (~CVs['High_DAPI']),'CV of SSC 95%'] = ub_2n
    CVs.loc[(CVs['Day']==day) & (~CVs['High_DAPI']),'CV of SSC 5%'] = lb_2n
    CVs.loc[(CVs['Day']==day) & (~CVs['High_DAPI']),'CV of SSC 95%'] = ub_2n
    CVs.loc[(CVs['Day']==day) & (CVs['High_DAPI']),'CV of SSC 5%'] = lb_4n
    CVs.loc[(CVs['Day']==day) & (CVs['High_DAPI']),'CV of SSC 95%'] = ub_4n
    
#%%

CV_2n = CVs[~CVs['High_DAPI']]
CV_4n = CVs[CVs['High_DAPI']]

plt.errorbar(np.arange(1,6),CV_2n['CV of FSC'],yerr=CV_2n['CV of FSC 95%']-CV_2n['CV of FSC 5%'])
plt.errorbar(np.arange(1,6),CV_4n['CV of FSC'],yerr=CV_4n['CV of FSC 95%']-CV_4n['CV of FSC 5%'])

plt.legend(['2n','4n'])
plt.xlabel('Days in culture')
plt.ylabel('CV of FSC')


CV_2n = CVs[~CVs['High_DAPI']]
CV_4n = CVs[CVs['High_DAPI']]

plt.figure()
plt.errorbar(np.arange(1,6),CV_2n['CV of SSC'],yerr=CV_2n['CV of SSC 95%']-CV_2n['CV of SSC 5%'])
plt.errorbar(np.arange(1,6),CV_4n['CV of SSC'],yerr=CV_4n['CV of SSC 95%']-CV_4n['CV of SSC 5%'])
plt.legend(['2n','4n'])
plt.xlabel('Days in culture')
plt.ylabel('CV of SSC')





