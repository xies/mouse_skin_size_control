#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:46:58 2020

@author: xies
"""

# Numerical / graphics
import pandas as pd
import numpy as np
from numpy import random
import seaborn as sb
import matplotlib.pylab as plt

# Utilities
from os import path
from glob import glob
import pickle
import copy
from tqdm import tqdm
from scipy import stats
# from mathUtils import cvariation_ci, cvariation_ci_bootstrap

import simulation

# Set random seed
np.random.seed(42)

# Growth rate is set to 0.01 per hour, i.e. 70hr doubling rate
max_iter = 400
dt = 1.0 # simulation step size in hours
# Total time simulated:
print(f'Total hrs simulated: {max_iter * dt / 70} generations')
Ncells = 1000

# Time information
sim_clock = {}
sim_clock['Max frame'] = max_iter
sim_clock['Max time'] = max_iter * dt
sim_clock['dt'] = dt

#%% Helper functions

def run_model(sim_clock, params, Ncells):
    
    # Seed initial cells asynchronously within the first 100h (~1.5 cell cycle times)
    async_birth_times = random.uniform(size=Ncells)*70
    
    initial_birth_times = dict(zip(range(Ncells),async_birth_times))
    
    population = dict()
    next_cellID = len(population)
    sim_clock['Current time'] = 0
    sim_clock['Current frame'] = 0
    
    for t in tqdm(np.arange(sim_clock['Max frame'] - 1)):
        
        # Check if any cells will be seeded this frame
        for k,v in initial_birth_times.items():
            if sim_clock['Current time'] >= v and sim_clock['Current time']-sim_clock['dt'] < v:
                # Seed cell into population
                # print(f'Seeding: {v}')
                new_seed = simulation.Cell(k,sim_clock,params)
                population[k] = new_seed
        
        # Advance time step by one
        sim_clock['Current frame'] += 1
        sim_clock['Current time'] += sim_clock['dt']
        
        newly_borns = {}
        for this_cell in population.values():
            
            # Skip cell if divided already
            if this_cell.divided:
                continue
            else:
                
                this_cell.advance_dt(sim_clock,params)
                
                if this_cell.divided:
                    
                    # Newly divided cell: make daughter cells
                    # print(f'\nCellID #{this_cell.cellID} has divided at frame {t}')
                    # Randomly draw an asymmettry
                    a = np.abs( random.randn(1)[0]*params['InhAsym']/100)
                    daughters = this_cell.divide(next_cellID, sim_clock, asymmetry=a)
                    next_cellID += 1
                    
                    newly_borns[daughters[0].cellID] = daughters[0]
                    # newly_borns[daughters[1].cellID] = daughters[1]
        
        population.update(newly_borns)
        
    return population

def plot_growth_curves_population(pop):
    
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
    
    CV_time = np.ones((max_iter,2))*np.nan
    for t in range(max_iter):
        p = phases[:,t]
        s = size[p == 'G1',t]
        if (len(s)>3):
            CV_time[t,0] = s.std()/s.mean()
        s = size[p == 'S/G2/M',t]
        if (len(s)>3):
            CV_time[t,1] = s.std()/s.mean()
    
    CV.loc['Population','G1'] = np.nanmean(CV_time[:,0])
    CV.loc['Population','S/G2/M'] = np.nanmean(CV_time[:,1])
    
    return CV

#%% Read parameter files

params = pd.read_csv('/Users/xies/OneDrive - Stanford/In vitro/CV from snapshot/CV model/params.csv',index_col=0)

#% Run model
runs = {}
CVs = {}
for model_name,p in params.iterrows():

    #% 1. Reset clock and initialize
    # Initialize each cell as a DataFrame at G1/S transition so we can specify Size and RB independently
    sim_clock['Current time'] = 0
    sim_clock['Current frame'] = 0
    
    # 2. Simulation steps
    population = run_model( sim_clock, p, Ncells)
    
    # 3. Collate data for analysis
    # Filter cells that have full cell cycles
    pop2analyze = {}
    for key,cell in population.items():
        if (cell.divided == True) & (cell.generation > 2):
            pop2analyze[key] = cell
    
    # plot_growth_curves_population(population)
    # Extract CVs
    CVs[model_name] = extract_CVs(pop2analyze)
    runs[model_name] = pop2analyze
    

#%%

CVs = {}
for model_name,pop2analyze in tqdm(runs.items()):
    CVs[model_name] = extract_CVs(pop2analyze,measurement_field='Volume')

CV_diff = []
method = []
name = []
for model_name,_df in CVs.items():
    
    CV_diff.append(_df.loc['Time','G1']- _df.loc['Time','S/G2/M'])
    method.append('Time')
    name.append(model_name)

    CV_diff.append(_df.loc['Population','G1']- _df.loc['Population','S/G2/M'])
    method.append('Population')
    name.append(model_name)
    
df = pd.DataFrame()
df['CV_diff'] = CV_diff
df['Method'] = method
df['model_name'] = name

 #%% size control graphs

pop2analyze = runs['adder_adder']

bsize = np.array([c.birth_size for c in pop2analyze.values()])
g1size = np.array([c.g1s_size for c in pop2analyze.values()])
dsize = np.array([c.div_size for c in pop2analyze.values()])
g1_growth = g1size - bsize
sg2_growth = dsize - g1size
total_growth = dsize - bsize

plt.figure()
plt.hist(bsize,50),plt.xlabel('Birth size'),plt.ylabel('G1| growth')

plt.figure()
plt.scatter(bsize,g1_growth),plt.xlabel('Birth size'),plt.ylabel('G1| growth')
plt.figure()
plt.scatter(g1size,sg2_growth),plt.xlabel('G1 size'),plt.ylabel('SG2M growth')
plt.figure()
plt.scatter(bsize,total_growth),plt.xlabel('Birth size'),plt.ylabel('Total growth')

#%% Cell cycle durations

Tg1 = np.array([cell.g1s_time - cell.ts['Time'].min() for cell in pop2analyze.values()])
Tdiv = np.array([cell.div_time - cell.ts['Time'].min() for cell in pop2analyze.values()])
Tsg2m = Tdiv - Tg1

plt.figure()
plt.subplot(1,3,1)
plt.hist(Tg1,50);plt.xlabel('G1 duration (h)')
plt.subplot(1,3,2)
plt.hist(Tsg2m,50);plt.xlabel('SG2M duration (h)')
plt.subplot(1,3,3)
plt.hist(Tdiv,50);plt.xlabel('Total duration (h)')

#%% CVs






#%% Boot strapped CIs

# Nboot = 1000
# collated['Measured volume'] = collated['Measured volume'].astype(float)

# def plot_size_CV_subplots(df,x,title=None):
    
#     plt.figure()
#     sb.barplot(df,y='Measured volume',x=x
#                 ,estimator=stats.variation,errorbar=(lambda x: cvariation_ci_bootstrap(x,Nboot)))
#     plt.ylabel('CV of Measured volume')
    
# plot_size_CV_subplots(collated,x='Phase',title='G1: sizer, SG2M: timer')


