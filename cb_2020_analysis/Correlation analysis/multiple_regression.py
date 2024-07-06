#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 16:31:07 2019

@author: xies
"""

import pandas as pd
import statsmodels.formula.api as smf


# Delete daughter cells from dfc
dfc = dfc[ dfc['Phase'] != 'Daughter G1']
dfc = dfc[ dfc['Phase'] != 'M']


# Rename variables for patsy
dfc['phase'] = pd.Categorical(dfc.Phase)
dfc = dfc.rename({'Frame':'frame'},axis=1)
dfc = dfc.rename({'Volume (sm)':'vol_sm'},axis=1)
dfc = dfc.rename({'Growth rate (sm)':'gr_sm'},axis=1)
dfc = dfc.rename({'Nucleus':'nuc'},axis=1)

# Growth rate regression: time v. phase v. current volume
est = smf.ols(formula="gr_sm ~ Age + vol_sm + Phase", data=dfc).fit()
est.summary()

est = smf.ols(formula="gr_sm ~ Age + vol_sm + Phase", data=dfc).fit()
est.summary()


# Growth rate regression: G1 cell only: early v late G1

# Curate early v late G1
g1_only = []
for c in collated_filtered:
    c = c[c['Phase'] == 'G1'].copy()
    c.iloc[0:3]['Phase'] = 'Early G1'
    c.iloc[3:]['Phase'] = 'Late G1'
    g1_only.append(c)

dfc = pd.concat(g1_only)
dfc = dfc.rename({'Frame':'frame'},axis=1)
dfc = dfc.rename({'Volume (sm)':'vol_sm'},axis=1)
dfc = dfc.rename({'Growth rate (sm)':'gr_sm'},axis=1)

est = smf.ols(formula="gr_sm ~  vol_sm + Phase", data=dfc).fit()
est.summary()


#
## Predict nuclear vol with cell phase or cell vol
#
## Curate early v late G1
#est = smf.ols(formula="gr_sm ~  vol_sm + Phase", data=dfc).fit()
#est.summary()
#
#



## NC ratio regression: time v. phase v. current volume
est = smf.ols(formula="nuc ~ phase + Volume", data=dfc).fit()
est.summary()
#
#est = smf.ols(formula="nuc ~ index + vol_sm", data=dfc).fit()
#est.summary()



##### Volume ~ Area + nuclear volume 
# NB: run crossarea_comaprison.py

df = pd.DataFrame()
df['Area'] = areas
df['Volume'] = volumes
df['Phase'] = phases
df['Nucleus'] = nuc

est = smf.ols(formula = 'Volume ~ Nucleus + Area',data=df).fit()
est.summary()




