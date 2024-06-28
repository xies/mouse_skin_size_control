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
import scipy as sp

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

        '''
        
        self.cellID = cellID
        # ts -- time series
        self.ts = pd.DataFrame( columns = ['Time','Volume','Phase'],
                            index=pd.RangeIndex( sim_clock['Max frame'] ), dtype=float)
        
        # scalar summaries of overall cell state (single statistics and not time-series)
        self.exp_growth_rate = np.nan
        self.birth_vol = np.nan
        self.div_vol = np.nan
        self.g1s_size = np.nan
        self.birth_time = np.nan
        self.div_time = np.nan
        self.g1s_time = np.nan
        self.birth_frame = np.nan
        self.div_frame = np.nan
        self.g1s_frame = np.nan
        
        # Flag for if cell is has already divided
        self.divided = False
        # @todo Store the parameters!
        
        # Check if a mother cell was passed in during construction
        if mother == None:
            # If Mother is not provided, must provide empirical parameters to initialize de novo
            # Will initialize a cell at birth
            assert(params is not None)
            # Create cell de novo using empirical paramters
            # Random log-normal birth sizes
            mu,sigma = _estimate_log_normal_parameters(params.loc['Volume']['BirthMean'],params.loc['Volume']['BirthStd'])
            birth_vol = random.lognormal(mean =mu, sigma=sigma)
            print(birth_vol)
            # Random normal exp growth rates
            gr = random.randn(1)*params.loc['Volume','GrStd'] + params.loc['Volume','GrMean']
            
            init_cell = pd.Series({'Time':sim_clock['Current time'],
                                   'Volume':birth_vol,'Phase':'G1'})
            self.parentID = np.nan
            self.birth_frame = sim_clock['Current frame']
            self.birth_time = sim_clock['Current time']
            self.birth_vol = birth_vol
            self.exp_growth_rate = gr
            
            self.generation = 0
            
        else: # Create daughter cell via symmetric dividion (ignores params)
            #NB: the "halving" is taken care of in @divide function
            init_vol = mother.div_vol * inheritance
            
            init_cell = pd.Series({'Time':sim_clock['Current time'],
                                   'Volume':init_vol,'Phase':'G1'})
            self.parentID = mother.cellID
            self.birth_vol = init_vol
            self.birth_frame = sim_clock['Current frame']
            self.birth_time = sim_clock['Current time']
            
            self.generation = mother.generation + 1
        
        self.ts.iloc[ sim_clock['Current frame'],:] = init_cell
        
    
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
            Dictionary containing current simulation time information, keyed by frame
        params : pd.Dataframe
            Empirical parameters
            
        Currently only uses Euler updates
             
        @todo: use Runge-Kutta for stability -- stick to Euler for now, num stability is no problem

        '''
        
        assert(self.divided == False) # sanity check that there is no division yet
        
        prev_frame = clock['Current frame'] - 1
        prev_values = self.ts.iloc[prev_frame] # pd.Series
        
        # Initialize current cell's values as pd.Series from prev. time point
        current_values = prev_values.copy()
        current_values['Time'] += clock['dt']
        
        # Grow cell in volume following parameter sheet
        dV = self.volume_growth(clock,params)
        current_values['Volume'] += dV
        
        # Check for G1/S transition
        if prev_values['Phase'] == 'G1' and self.g1s_transition(clock,params):
            current_values['Phase'] = 'S/G2/M'
            self.g1s_time = current_values['Time']
            self.g1s_size = current_values['Volume']
            self.g1s_frame = clock['Current frame']
            
        # Check for division
        if prev_values['Phase'] == 'S/G2/M' and self.decide2divide(clock,params):
            
            current_values['Volume'] = np.nan
            current_values['Phase'] = 'None'
            self.div_time = prev_values['Time']
            self.div_vol = prev_values['Volume']
            self.div_frame = prev_frame
            self.divided = True
        
        # Put the current values into the timeseries
        self.ts.iloc[prev_frame + 1,:] = current_values
        
    
    def volume_growth(self, sim_clock, params):
        frame = sim_clock['Current frame'] - 1
        # Simple exponential growth
        cell = self.ts.iloc[frame]
        
        dV = cell['Volume'] * self.exp_growth_rate # Euler update
        # Assume no measurement noise for now
        # noise = random.randn() * params['Size']['GrStd'
        total = dV
        return total * sim_clock['dt']
        
    def g1s_transition(self,sim_clock,params):
        # Always 'work' based on prev_frame information
        frame = sim_clock['Current frame'] -1
        cell = self.ts.iloc[frame]
        theta = params.loc['Volume','G1S_sizethreshold']
        if cell['Volume'] > theta:
            return True
        else:
            return False
        
    
    def decide2divide(self,sim_clock,params):
        # Always 'work' based on prev_frame information
        frame = sim_clock['Current frame'] - 1
        # Decide to divide if S/G2/M duration has lasted a certain time duration (following empirical parameter sheet)
        # @todo: use Metropolis sampling
        
        cell = self.ts.iloc[frame]
        # Only S/G2/M cells divide
        assert(cell['Phase'] == 'S/G2/M')
        
        # Calculate Tsg2m
        time_of_g1s = self.g1s_time
        Tsg2m = sim_clock['Current time'] - time_of_g1s
        
        # Construct timing distrubtion LUT
        Tsg2m_distribution = sp.stats.norm(loc=params['Tsg2mMean'],scale=params['Tsg2mStd'])
        
        prob_threshold = Tsg2m_distribution.pdf(Tsg2m)
        
        # Metropolis method
        p = random.uniform(size=1)[0]
        if p < prob_threshold:
            return True
        else:
            return False 
       
#--- helper functions ---
    @staticmethod
    def _estimate_log_normal_parameters(sample_mu,sample_sigma):
        '''
        Convert a sample mean and std from a log-normal distr. and convert to the underlying
        normal distr mu and sigma
        
        Parameters
        ----------
        sample_mu : TYPE
            DESCRIPTION.
        sample_sigma : TYPE
            DESCRIPTION.

        Returns
        -------
        mu, sigma of 'not-log' normal distribution
        
        See: https://en.wikipedia.org/wiki/Log-normal_distribution

        '''
        mu = np.log(sample_mu**2 / np.sqrt(sample_mu**2+sample_sigma**2))
        sigma = np.sqrt( np.log(1+sample_sigma**2/sample_mu**2) )        
        return mu,sigma
        
        
# --- DATA METHODS ----
    # def decatenate_fields()
    def __repr__(self):
        string = 'Cell ID = ' + str(self.cellID) + '\n'
        string += 'Born at frame ' + str(self.birth_frame) + '\n'
        string += 'Divided : ' + str(self.divided) + '\n\n\n'
        return string

