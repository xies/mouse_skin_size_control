#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:44:12 2024

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mathUtils import cvariation_bootstrap, cv_difference_pvalue
from os import path

# Load all the datasets
dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/zebrafish_ditalia/osx_fucci_26hpp_11_4_17/'
df = pd.read_csv(path.join(dirname,'cell_size_by_cellcycle_position.csv'),index_col=0)

