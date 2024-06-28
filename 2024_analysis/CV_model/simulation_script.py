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

# Growth rate is set to 0.01, i.e. 70hr doubling rate
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

simulation_dir = '/Users/xies/Box/Bioinformatics/scaling_model/'
project_dir = 'synth_slope_1/vary_deg_rate/'

parameter_settings = {}
for pfile in glob(path.join(simulation_dir,project_dir,'*.csv')):
    parameter_settings[pfile] = pd.read_csv(pfile, index_col = 0)

#%% Run model 

# synthesis_slopes = [-1,-0.5,-0.25,0,0.25,0.5,1,1.5]
# synthesis_thetas = [-np.pi/6,0,np.pi/6,np.pi/4, np.pi/3]
theta = np.pi/3
deg_rates = [1,0.5,0.3,0.2,0.1,0.08,0.05,0.02,0.01,0.005]
size_bins = np.linspace(1.2,4,40)

steady_state_scaling = np.zeros((len(deg_rates),len(size_bins) - 1))
bin_centers = (size_bins[0:-1] + size_bins[1:])/2

for pfile,params in parameter_settings.items():
    
    subdir = path.dirname(pfile)
    mean_turnover = np.zeros(len(deg_rates))
    synthesis_slopes = []
    for i,k in enumerate(deg_rates):
    
        # 1. Set the flexible parameter
        params.at['Protein','base synthesis rate'] = 0.1
        params.at['Protein','degradation'] = k
        # if doing linear scaling, calculate scaling relations
        slope, intercept = synthesis_scaling_rotation(theta)
        slope  = 0.9
        intercept = 0.1
        params.at['Protein','scaling slope'] = slope
        params.at['Protein','scaling intercept'] = intercept
        synthesis_slopes.append(slope)
        
        #% 2. Reset clock and initialize
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
        
        # Retrieve each datafield into dataframe
        time = np.vstack( [ cell.ts['Time'].astype(np.float) for cell in pop2analyze.values() ])
        size = np.vstack( [ cell.ts['Volume'].astype(np.float) for cell in pop2analyze.values() ])
        # prot = np.vstack( [ cell.ts['Protein'].astype(np.float) for cell in pop2analyze.values() ])
        protein_conc = np.vstack( [ cell.ts['Protein conc'].astype(np.float) for cell in pop2analyze.values() ])
        protein_turnover = np.vstack( [ cell.ts['Protein frac turnover'].astype(np.float) for cell in pop2analyze.values() ])
        
        # Get mean steady state size v concentration plots
        means = get_bin_means(size,protein_conc,bin_edges= size_bins)
        steady_state_scaling[i,:] = standardize(means)
        
        # 5. Save individual runs 
        with open(path.join(subdir,f'model_slope_{slope}.pkl'),'wb') as f:
              pickle.dump([initial_pop,population], f)
    
        # get average turnover per condition
        mean_turnover[i] = np.nanmean(protein_turnover)
    
    # # Fit lines to steady state slopes
    # ss_slopes = np.zeros(len(deg_rates))
    # for i in range(len(deg_rates)):
    #     # Filter out NaNs from pair
    #     X = bin_centers; Y = steady_state_scaling[i,:]
    #     X,Y = nonan_pairs(X,Y)
    #     p = np.polyfit(X,Y,1)
    #     ss_slopes[i] = p[0]
    
    # Save steady state size-dynamics v. synthesis slopes for this sweep
    plt.plot(bin_centers.T,steady_state_scaling.T);
    plt.xlabel('Cell size');plt.ylabel('Normalized protein conc');plt.legend(deg_rates)
    alpha = params.at['Protein','base synthesis rate']
    beta = params.at['Protein','degradation']
    slope = params.at['Protein','scaling slope']
    # plt.title(f'Turnoever = {mean_turnover.mean()}; alpha = {alpha}; deg_rate = {beta}')
    plt.title(f'synthesis scaling = {slope}; alpha = {alpha};')
    plt.savefig( path.join(subdir, 'steady_state_scaling.png') )
    plt.close()
    
    
    print(f'Done with {pfile}')
    
    


