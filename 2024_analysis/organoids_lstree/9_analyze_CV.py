#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:09:52 2024

@author: xies
"""


import pandas as pd
import numpy as np
from os import path
import matplotlib.pyplot as plt
from mathUtils import cvariation_bootstrap, cv_difference_pvalue

summary = pd.read_csvpath.join('/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/size_summary.csv'),index_col=0)

#%% Print CV by cell cycle phase

#@ need more division for CV

CV = pd.DataFrame()

CV.loc['Birth',['CV','LB','UB']] = cvariation_bootstrap(summary['Birth volume'],Nboot=1000,subsample=80)
CV.loc['G1S',['CV','LB','UB']] = cvariation_bootstrap(summary['G1 volume'],Nboot=1000,subsample=80)
# CV.loc['Division',['CV','LB','UB']] = cvariation_bootstrap(summary['Division volume'],Nboot=1000,subsample=80)
