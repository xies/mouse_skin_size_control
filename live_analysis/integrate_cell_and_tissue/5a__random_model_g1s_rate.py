#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 18:25:36 2022

@author: xies
"""

import numpy as np
from numpy import random
import statsmodels.formula.api as smf

import pandas as pd

#%%

N = 500

X1 = random.randn(N)
X2 = X1 + random.randn(N) * 0.1
X3 = random.randn(N)
y = X1 + random.randn(N)*0.1


df = pd.DataFrame()
df['y'] = y
df['X1'] = X1
df['X2'] = X2
df['X3'] = X3

model = smf.rlm('y ~ X1+X2+X3',data=df).fit()
model.summary()

