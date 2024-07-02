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

import simulation

# Set random seed
np.random.seed(42)

# Growth rate is set to 0.01 per hour, i.e. 70hr doubling rate
max_iter = 300
dt = 12 # simulation step size in hours
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
                    print(f'\nCellID #{this_cell.cellID} has divided at frame {t}')
                    # Randomly draw an asymmettry
                    a = np.abs( random.randn(1)[0]*0.05)
                    daughters = this_cell.divide(next_cellID, sim_clock, asymmetry=a)
                    next_cellID += 1
                    
                    newly_borns[daughters[0].cellID] = daughters[0]
                    # newly_borns[daughters[1].cellID] = daughters[1]
        
        population.update(newly_borns)
        
    return population

#%% Read parameter files

params = pd.read_csv('/Users/xies/OneDrive - Stanford/In vitro/CV from snapshot/CV model/params.csv',index_col=0)

#% Run model 

#% 1. Reset clock and initialize
# Initialize each cell as a DataFrame at G1/S transition so we can specify Size and RB independently
sim_clock['Current time'] = 0
sim_clock['Current frame'] = 0

# initial_pop = initialize_model(params, sim_clock, Ncells)

# 2. Simulation steps
population = run_model( sim_clock, params, Ncells)

# 3. Collate data for analysis
# Filter cells that have full cell cycles
pop2analyze = {}
for key,cell in population.items():
    if (cell.divided == True and cell.generation > 2):
        pop2analyze[key] = cell

#%% Clean up the dataframes

collated = []
for key,cell in pop2analyze.items():
    ts = cell.ts.dropna()
    ts.loc[:,'Age'] = ts.loc[:,'Time'] - ts.iloc[0]['Time']
    collated.append(ts)

CV = pd.DataFrame()
for phase,x in collated[0].groupby('Phase')['Measured volume']:
    CV.loc['Time',phase] = x.std()/x.mean()

Tg1 = np.array([cell.g1s_time - cell.ts['Time'].min() for cell in pop2analyze.values()])
Tdiv = np.array([cell.div_time - cell.ts['Time'].min() for cell in pop2analyze.values()])
Tsg2m = Tdiv - Tg1

#%% # Retrieve each datafield into a time-slice

time = np.vstack( [ cell.ts['Time'].astype(float) for cell in pop2analyze.values() ])
size = np.vstack( [ cell.ts['Measured volume'].astype(float) for cell in pop2analyze.values() ])
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

#%%

def plot_growth_curves_population(pop):
    
    for cell in pop.values():
        ts = cell.ts.dropna()
        t = ts['Time']
        v = ts['Measured volume']
        p = ts['Phase']
        
        t_g1 = t[p =='G1']
        v_g1 = v[p =='G1']
        plt.plot(t_g1,v_g1,'b-',alpha=0.1)
        
        t_g2 = t[p =='S/G2/M']
        v_g2 = v[p =='S/G2/M']
        plt.plot(t_g2,v_g2,'r-',alpha=0.1)

plot_growth_curves_population(pop2analyze)

