#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:45:04 2020

@author: xies
"""


# @todo: add synth + degradation of protein species


import numpy as np
from numpy import random
import pandas as pd

class Cell():
    
    def __init__(self, cellID, sim_clock, params=None, mother=None, inheritance=0.5):
        '''
        Cell for simulating cell growth/division coupling

        Parameters
        ----------
        cellID : int
            Unique cell ID
        sim_clock : dict
            Dictionary containing the time information of simulation
        params : pd.DataFrame
            empirical parameters extracted from DFB dataset
        mother : Cell, optional
            If inheriting properties from a mother. The default is None.
        inheritance : TYPE, optional
            fraction of inheritance from mother. The default is 0.5.

        Returns
        -------
        Cell object.
        
        
        Definitions
        -----------
        
        Volume = cell volume
        Protein = Number of protein
        Protein concentration = Protein / Volume

        '''
        
        self.cellID = cellID
        # ts -- time series
        self.ts = pd.DataFrame( columns = ['Time','Volume','Protein','Protein conc','Protein frac turnover','Phase'],
                            index=pd.RangeIndex( sim_clock['Max frame'] ), dtype=np.float)
        # scalar summaries of cell state (snapshots/single statistics)
        self.birth_vol = np.nan
        self.div_vol = np.nan
        # self.g1s_size = np.nan
        self.birth_time = np.nan
        self.div_time = np.nan
        # self.g1s_time = np.nan
        self.birth_frame = np.nan
        self.div_frame = np.nan
        # self.g1s_frame = np.nan
        self.birth_protein = np.nan
        # self.g1s_protein = np.nan
        self.div_protein = np.nan
        self.birth_protein_conc = np.nan
        # self.g1s_rb_conc = np.nan
        self.div_protein_conc = np.nan
        # Flag for if cell is has already divided
        self.divided = False
        # Store the parameters!
        
        # Check if a mother cell was passed in during construction
        if mother == None:
            # If Mother is not provided, must provide empirical parameters to initialize de novo
            # Will initialize a cell at birth
            assert(params is not None)
            # Create cell de novo using empirical paramters
            # g1s_size = random.lognormal(mean = params.loc['Size','Mean G1S'],sigma = params.loc['Size','Std G1S'])
            # g1s_rb = random.lognormal(mean = params.loc['RB','Mean G1S'],sigma = params.loc['RB','Std G1S'])
            # rb_conc = g1s_rb / g1s_size
            
            # birth_vol = random.lognormal(mean = params.loc['Volume']['Mean Birth'], sigma = params.loc['Volume']['Std Birth'])
            # birth_protein = random.lognormal(mean = params.loc['Protein']['Mean Birth'], sigma = params.loc['Protein']['Std Birth'])
            birth_vol = 1
            # set birth protein to steady-state prediction
            birth_protein = params.loc['Protein','base synthesis rate'] / params.loc['Protein','degradation']
            protein_conc = birth_protein / birth_vol
            
            init_cell = pd.Series({'Time':sim_clock['Current time'],
                                   'Volume':birth_vol,'Protein':birth_protein,'Protein conc':protein_conc})
            self.parentID = np.nan
            self.birth_frame = sim_clock['Current frame']
            self.birth_time = sim_clock['Current time']
            self.birth_vol = birth_vol
            self.birth_protein = birth_protein
            self.birth_protein_conc = protein_conc
            
            self.generation = 0
            
        else: # Create daughter cell via symmetric dividion (ignores params)
            #NB: the "halving" is taken care of in @divide function
            init_vol = mother.div_vol * inheritance
            init_protein = mother.div_protein * inheritance
            protein_conc = init_protein / init_vol
            
            init_cell = pd.Series({'Time':sim_clock['Current time'],
                                   'Volume':init_vol,'Protein':init_protein,'Protein conc':protein_conc})
            self.parentID = mother.cellID
            self.birth_vol = init_vol
            self.birth_frame = sim_clock['Current frame']
            self.birth_time = sim_clock['Current time']
            self.birth_protein = init_protein
            self.birth_protein_conc = protein_conc
            
            self.generation = mother.generation + 1
        
        self.ts.at[ sim_clock['Current frame'] ] = init_cell
        
    
    def divide( self, cellID_beginning, sim_clock, asymmetry = 0.0):
        '''
        Divides the current mother cell into two daughters.

        Parameters
        ----------
        cellID_beginning : int
            the cellID of first daughter, +1 will be second daughter
        sim_clock : dict
            Dictionary containing the time information of simulation
        asymmetry : float, optional
            The difference between larger daughter + smaller daughter normalized by mother.
            sym = (D_L - D_s) / M
            The default is 0.0.

        Returns
        -------
        daughter_a : Cell
            Larger daughter
        daughter_b : Cell
            Smaller daughter

        '''
        current_frame = sim_clock['Current frame']
        
        # Onlly allow G2 divisions
        # assert(self.ts.iloc[current_frame]['Phase'] == 'None')
        
        # Calculate respective inheritance fractions 
        assert(asymmetry < 1.0)
        inh_a = (asymmetry + 1.) / 2
        inh_b = 1 - (asymmetry + 1.) / 2
        
        daughter_a = Cell(cellID_beginning, sim_clock, mother=self, inheritance=inh_a)
        daughter_b = Cell(cellID_beginning+1, sim_clock, mother=self, inheritance=inh_b)
        
        return (daughter_a, daughter_b)

# --- SIMULATION METHODS ----

    def advance_dt(self,clock,params):
        '''
        
        Parameters
        ----------
        clock : dict
            Dictionary containing current simulation time information
        params : pd.Dataframe
            Empirical parameters extracted from DFB
            
        Currently only uses Euler updates
             
        @todo: use Runge-Kutta for stability -- stick to Euler for now, num stability is no problem

        '''
        
        assert(self.divided == False)
        
        prev_frame = clock['Current frame'] - 1
        prev_values = self.ts.iloc[prev_frame] # pd.Series
        
        # Initialize current cell's values as pd.Series from prev. time point
        current_values = prev_values.copy()
        current_values['Time'] += clock['dt']
        
        # First calculate growths based on previous values
        #
        # dV/dt = growth_rate * V(t)
        #
        dV = self.volume_growth(clock,params)
        
        # Then protein based on current values
        #
        # Protein => # of protein molecules in cell
        # dP(t)/dt = (1 + size_scaling_slope) * base_synthesis_rate * V(t) - deg_rate * P(t) / V(t)
        #
        prot_synthesis = self.protein_synthesis(clock,params)
        prot_deg = self.protein_degradation(clock,params) # should already be negative
        
        dprotein = prot_synthesis + prot_deg
        
        # Add to current values
        current_values['Volume'] += dV
        current_values['Protein frac turnover'] = (abs(prot_synthesis) + abs(prot_deg)) / current_values['Protein']
        current_values['Protein'] += dprotein
        current_values['Protein conc'] = current_values['Protein'] / current_values['Volume']
        
        # Check for division, if so, reset cell
        if self.decide2divide(clock,params):
            current_values['Protein'] = np.nan
            current_values['Volume'] = np.nan
            current_values['Protein conc'] = np.nan
            current_values['Protein frac turnover'] = np.nan
            current_values['Phase'] = 'None'
            self.div_time = prev_values['Time']
            self.div_vol = prev_values['Volume']
            self.div_frame = prev_frame
            self.div_protein = prev_values['Protein']
            self.div_protein_conc = prev_values['Protein conc']
            self.divided = True
        
        # Put the current values into the timeseries
        self.ts.at[prev_frame + 1] = current_values
        
    def protein_degradation(self, sim_clock, params):

        frame = sim_clock['Current frame'] - 1
        #First order equation
        k_deg = params.loc['Protein','degradation']
        dprotein = -k_deg * self.ts.iloc[frame]['Protein']
        
        return dprotein * sim_clock['dt']
    
    
    def protein_synthesis(self, sim_clock, params):
        frame = sim_clock['Current frame'] - 1
        # dP/dt
        cell = self.ts.iloc[frame]
        
        # synthesis using normal centered at size = 1.5
        # synthesis = params.loc['Protein','base synthesis rate'] * np.exp(-(cell['Volume'] - 1.5)**2 / 0.3)
        
        slope = params.loc['Protein','scaling slope']
        intercept = params.loc['Protein','scaling intercept']
        
        # slope,intercept =(1.5 - 3*slope) / 2. # keep size-average to be the same
        # intercept = 2
        
        synthesis = params.loc['Protein','base synthesis rate'] * (intercept + slope * cell['Volume'])
        
        sigma = 0 # Noise term
        total = synthesis + random.randn() * sigma
        
        return (total) * sim_clock['dt']
    
    
    def volume_growth(self, sim_clock, params):
        frame = sim_clock['Current frame'] - 1
        # Simple exponential growth
        cell = self.ts.iloc[frame]
        
        # dV = params.loc['Volume','exponential growth rate']
        dV = cell['Volume'] * params.loc['Volume','exponential growth rate']
        # noise = random.randn() * params['Size']['growth noise']
        noise = 0
        total = dV + noise
        return total * sim_clock['dt']
        
    
    def decide2divide(self,sim_clock,params):
        frame = sim_clock['Current frame'] - 1
        # Decide if cells has doubled in size
        
        cell = self.ts.iloc[frame]
        
        if cell['Volume'] > 2:
            return True
        else:
            return False 
        
# --- DATA METHODS ----
    # def decatenate_fields()
    def __repr__(self):
        string = 'Cell ID = ' + str(self.cellID) + '\n'
        string += 'Born at frame ' + str(self.birth_frame) + '\n'
        string += 'Divided : ' + str(self.divided) + '\n\n\n'
        return string

