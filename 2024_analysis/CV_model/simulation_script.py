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
max_iter = 9000
dt = 10.0/60 # simulation step size in hours
# Total time simulated:
print(f'Total hrs simulated: {max_iter * dt / 70} generations or {2**(max_iter * dt / 70)} cells')
Ncells = 1

# Time information
sim_clock = {}
sim_clock['Max frame'] = max_iter
sim_clock['Max time'] = max_iter * dt
sim_clock['dt'] = dt

#%% Helper functions

def initialize_model(params, sim_clock, Ncells):
    next_cellID = 0
    initial_pop = {}
    for i in range(Ncells):
        # Initialize cells de novo
        cell = simulation.Cell(i, sim_clock, params)
        initial_pop[i] = cell
        next_cellID += 1
    return initial_pop

def run_model(initial_pop, sim_clock, params):
    
    next_cellID = len(initial_pop)
    population = copy.deepcopy(initial_pop)
    sim_clock['Current time'] = 0
    sim_clock['Current frame'] = 0
    
    for t in tqdm(np.arange(sim_clock['Max frame'] - 1)):
        
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
                    print(f'CellID #{this_cell.cellID} has divided at frame {t}')
                    # Randomly draw an asymmettry
                    a = np.abs( random.randn() * 0.00 ) # 5 percent
                    daughters = this_cell.divide(next_cellID, sim_clock, asymmetry=a)
                    next_cellID += 2
                    # Put only one daughter into the population (no growth)
                    newly_borns[daughters[0].cellID] = daughters[0]
                    # newly_borns[daughters[1].cellID] = daughters[1]
        
        population.update(newly_borns)
        
    return population

#%% Read parameter files

params = pd.read_csv('/Users/xies/OneDrive - Stanford/In vitro/CV from snapshot/CV model/params.csv',index_col=0)

#%% Run model 

#% 1. Reset clock and initialize
# Initialize each cell as a DataFrame at G1/S transition so we can specify Size and RB independently
sim_clock['Current time'] = 0
sim_clock['Current frame'] = 0

initial_pop = initialize_model(params, sim_clock, Ncells)

# 3. Simulation steps
population = run_model(initial_pop, sim_clock, params)

# 4. Collate data for analysis
# Filter cells that have full cell cycles
pop2analyze = {}
for key,cell in population.items():
    if (cell.divided == True) & (cell.generation > 4):
        pop2analyze[key] = cell

#%% Clean up the dataframes

collated = []
for key,cell in population.items():
    ts = cell.ts.dropna()
    ts['Age'] = ts['Time'] - ts.iloc[0]['Time']
    collated.append(ts)
    
    

# Retrieve each datafield into dataframe
time = np.vstack( [ cell.ts['Time'].astype(float) for cell in pop2analyze.values() ])
size = np.vstack( [ cell.ts['Volume'].astype(float) for cell in pop2analyze.values() ])

size = np.vstack( [ cell.ts['Volume'].astype(float) for cell in pop2analyze.values() ])

# 5. Save individual runs 
# with open(path.join(subdir,f'model_slope_{slope}.pkl'),'wb') as f:
#       pickle.dump([initial_pop,population], f)


