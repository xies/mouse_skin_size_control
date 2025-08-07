#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 12:04:02 2021

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io, measure
from os import path
import seaborn as sb
from scipy.stats import stats
import matplotlib.pyplot as plt

from matplotlib.path import Path
from SelectFromCollection import SelectFromCollection


dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/RBKO p107KO/M3 DOB 08-20-2023/11-07-2023 DKO ear (DOB 08-20-23, tam)/M3 p107homo Rbfl'

dx = 0.2920097

#%% Load segmentations
#NB: In variable name it's called DMSO but it's actually ethanol control

seg_tam_before_r1 = io.imread(path.join(dirname,'Left ear/Post tam/R1/cellpose_basal_layer_cleaned_byhand','t0.tif'))
seg_tam_after_r1 = io.imread(path.join(dirname,'Left ear/Post tam/R1/cellpose_basal_layer_cleaned_byhand','t10.tif'))
h2b_tam_before_r1 = io.imread(path.join(dirname,'Left ear/Post tam/R1/10. Day 5 (0)','G_reg.tif'))
h2b_tam_after_r1 = io.imread(path.join(dirname,'Left ear/Post tam/R1/20. Day 10 (10)','B_align.tif'))
fucci_tam_before_r1 = io.imread(path.join(dirname,'Left ear/Post tam/R1/10. Day 5 (0)','R_reg_reg.tif'))
fucci_tam_after_r1 = io.imread(path.join(dirname,'Left ear/Post tam/R1/20. Day 10 (10)','R_align.tif'))

seg_tam_before_r2 = io.imread(path.join(dirname,'Left ear/Post tam/R2/cellpose_basal_layer_cleanedbyhand','t0.tif'))
seg_tam_after_r2 = io.imread(path.join(dirname,'Left ear/Post tam/R2/cellpose_basal_layer_cleanedbyhand','t11.tif'))
h2b_tam_before_r2 = io.imread(path.join(dirname,'Left ear/Post tam/R2/10. Day 5','G_reg.tif'))
h2b_tam_after_r2 = io.imread(path.join(dirname,'Left ear/Post tam/R2/21. Day 10.5','G_align.tif'))
fucci_tam_before_r2 = io.imread(path.join(dirname,'Left ear/Post tam/R2/10. Day 5','R_reg_reg.tif'))
fucci_tam_after_r2 = io.imread(path.join(dirname,'Left ear/Post tam/R2/21. Day 10.5','R_align.tif'))

seg_tam_before_r5 = io.imread(path.join(dirname,'Left ear/Post tam/R5/cellpose_basal_layer_byhand','t0.tif'))
seg_tam_after_r5 = io.imread(path.join(dirname,'Left ear/Post tam/R5/cellpose_basal_layer_byhand','t13.tif'))
h2b_tam_before_r5 = io.imread(path.join(dirname,'Left ear/Post tam/R5/10. Day 5','G_reg.tif'))
h2b_tam_after_r5 = io.imread(path.join(dirname,'Left ear/Post tam/R5/23. Day 11.5','G_reg.tif'))
fucci_tam_before_r5 = io.imread(path.join(dirname,'Left ear/Post tam/R5/10. Day 5','R_reg_reg.tif'))
fucci_tam_after_r5 = io.imread(path.join(dirname,'Left ear/Post tam/R5/23. Day 11.5','R_reg_reg.tif'))

seg_tam_before_r6 = io.imread(path.join(dirname,'Left ear/Post tam/R6/cellpose_basal_layer_cleaned_byhand','t1.tif'))
seg_tam_after_r6 = io.imread(path.join(dirname,'Left ear/Post tam/R6/cellpose_basal_layer_cleaned_byhand','t13.tif'))
h2b_tam_before_r6 = io.imread(path.join(dirname,'Left ear/Post tam/R6/11. Day 5.5','G_reg.tif'))
h2b_tam_after_r6 = io.imread(path.join(dirname,'Left ear/Post tam/R6/23. Day 11.5','G_reg.tif'))
fucci_tam_before_r6 = io.imread(path.join(dirname,'Left ear/Post tam/R6/11. Day 5.5','R_reg_reg.tif'))
fucci_tam_after_r6 = io.imread(path.join(dirname,'Left ear/Post tam/R6/23. Day 11.5','R_reg_reg.tif'))

seg_dmso_before_r1 = io.imread(path.join(dirname,'Right ear/Post Ethanol/R1/cellpose_G_clahe_basal_byhand','t0.tif'))
seg_dmso_after_r1 = io.imread(path.join(dirname,'Right ear/Post Ethanol/R1/cellpose_G_clahe_basal_byhand','t10.tif'))
h2b_dmso_before_r1 = io.imread(path.join(dirname,'Right ear/Post Ethanol/R1/10. Day 5 (0)','G_reg.tif'))
h2b_dmso_after_r1 = io.imread(path.join(dirname,'Right ear/Post Ethanol/R1/22. Day 11 (12)','G_align.tif'))
fucci_dmso_before_r1 = io.imread(path.join(dirname,'Right ear/Post Ethanol/R1/10. Day 5 (0)','R_reg_reg.tif'))
fucci_dmso_after_r1 = io.imread(path.join(dirname,'Right ear/Post Ethanol/R1/22. Day 11 (12)','R_align.tif'))

seg_dmso_before_r2 = io.imread(path.join(dirname,'Right ear/Post Ethanol/R2/cellpose_basal_layer_cleaned_byhand','t0.tif'))
seg_dmso_after_r2 = io.imread(path.join(dirname,'Right ear/Post Ethanol/R2/cellpose_basal_layer_cleaned_byhand','t10.tif'))
h2b_dmso_before_r2 = io.imread(path.join(dirname,'Right ear/Post Ethanol/R2/10. Day 5','G_reg.tif'))
h2b_dmso_after_r2 = io.imread(path.join(dirname,'Right ear/Post Ethanol/R2/20. Day 10','G_align.tif'))
fucci_dmso_before_r2 = io.imread(path.join(dirname,'Right ear/Post Ethanol/R2/10. Day 5','R_reg_reg.tif'))
fucci_dmso_after_r2 = io.imread(path.join(dirname,'Right ear/Post Ethanol/R2/20. Day 10','R_align.tif'))

seg_dmso_before_r3 = io.imread(path.join(dirname,'Right ear/Post Ethanol/R3/cellpose_basal_layer_cleaned_byhand','t0.tif'))
seg_dmso_after_r3 = io.imread(path.join(dirname,'Right ear/Post Ethanol/R3/cellpose_basal_layer_cleaned_byhand','t10.tif'))
h2b_dmso_before_r3 = io.imread(path.join(dirname,'Right ear/Post Ethanol/R3/10. Day 5','G_reg.tif'))
h2b_dmso_after_r3 = io.imread(path.join(dirname,'Right ear/Post Ethanol/R3/20. Day 10','G_align.tif'))
fucci_dmso_before_r3 = io.imread(path.join(dirname,'Right ear/Post Ethanol/R3/10. Day 5','R_reg_reg.tif'))
fucci_dmso_after_r3 = io.imread(path.join(dirname,'Right ear/Post Ethanol/R3/20. Day 10','R_align.tif'))

#%%

def measure_intensity_two_channels(seg,h2b_chan,fucci_chan):
    df1 = pd.DataFrame(measure.regionprops_table(seg,h2b_chan,properties=['label','area','mean_intensity']))
    df1 = df1.rename(columns={'mean_intensity':'H2b'})
    df2 = pd.DataFrame(measure.regionprops_table(seg,fucci_chan,properties=['label','area','mean_intensity']))
    df2 = df2.rename(columns={'mean_intensity':'FUCCI'})
    df2 = df2.drop(columns=['area'])
    df = pd.merge(df1,df2,on='label')
    return df

tam_before = measure_intensity_two_channels(seg_tam_before_r1,h2b_tam_before_r1,fucci_tam_before_r1)
tam_before['Genotype'] = '4OHT'
tam_before['Region'] = '4OHT R1'
tam_before['Time'] = 0
tam_after = measure_intensity_two_channels(seg_tam_after_r1,h2b_tam_after_r1,fucci_tam_after_r1)
tam_after['Genotype'] = '4OHT'
tam_after['Region'] = '4OHT R1'
tam_after['Time'] = 1
tam = pd.concat((tam_before,tam_after),ignore_index=True)

tam_before = measure_intensity_two_channels(seg_tam_before_r2,h2b_tam_before_r2,fucci_tam_before_r2)
tam_before['Genotype'] = '4OHT'
tam_before['Region'] = '4OHT R2'
tam_before['Time'] = 0
tam_after = measure_intensity_two_channels(seg_tam_after_r2,h2b_tam_after_r2,fucci_tam_after_r2)
tam_after['Genotype'] = '4OHT'
tam_after['Region'] = '4OHT R2'
tam_after['Time'] = 1
tam = pd.concat((tam,tam_before,tam_after),ignore_index=True)

tam_before = measure_intensity_two_channels(seg_tam_before_r5,h2b_tam_before_r5,fucci_tam_before_r5)
tam_before['Genotype'] = '4OHT'
tam_before['Region'] = '4OHT R5'
tam_before['Time'] = 0
tam_after = measure_intensity_two_channels(seg_tam_after_r5,h2b_tam_after_r5,fucci_tam_after_r5)
tam_after['Genotype'] = '4OHT'
tam_after['Region'] = '4OHT R5'
tam_after['Time'] = 1
tam = pd.concat((tam,tam_before,tam_after),ignore_index=True)

tam_before = measure_intensity_two_channels(seg_tam_before_r6,h2b_tam_before_r6,fucci_tam_before_r6)
tam_before['Genotype'] = '4OHT'
tam_before['Region'] = '4OHT R6'
tam_before['Time'] = 0
tam_after = measure_intensity_two_channels(seg_tam_after_r6,h2b_tam_after_r6,fucci_tam_after_r6)
tam_after['Genotype'] = '4OHT'
tam_after['Region'] = '4OHT R6'
tam_after['Time'] = 1
tam = pd.concat((tam,tam_before,tam_after),ignore_index=True)

dmso_before = measure_intensity_two_channels(seg_dmso_before_r1,h2b_dmso_before_r1,fucci_dmso_before_r1)
dmso_before['Genotype'] = 'DMSO'
dmso_before['Region'] = 'DMSO R1'
dmso_before['Time'] = 0
dmso_after = measure_intensity_two_channels(seg_dmso_after_r1,h2b_dmso_after_r1,fucci_dmso_after_r1)
dmso_after['Genotype'] = 'DMSO'
dmso_after['Region'] = 'DMSO R1'
dmso_after['Time'] = 1
dmso = pd.concat((dmso_before,dmso_after),ignore_index=True)

dmso_before = measure_intensity_two_channels(seg_dmso_before_r2,h2b_dmso_before_r2,fucci_dmso_before_r2)
dmso_before['Genotype'] = 'DMSO'
dmso_before['Region'] = 'DMSO R2'
dmso_before['Time'] = 0
dmso_after = measure_intensity_two_channels(seg_dmso_after_r2,h2b_dmso_after_r2,fucci_dmso_after_r2)
dmso_after['Genotype'] = 'DMSO'
dmso_after['Region'] = 'DMSO R2'
dmso_after['Time'] = 1
dmso = pd.concat((dmso,dmso_before,dmso_after),ignore_index=True)

dmso_before = measure_intensity_two_channels(seg_dmso_before_r3,h2b_dmso_before_r3,fucci_dmso_before_r3)
dmso_before['Genotype'] = 'DMSO'
dmso_before['Region'] = 'DMSO R3'
dmso_before['Time'] = 0
dmso_after = measure_intensity_two_channels(seg_dmso_after_r3,h2b_dmso_after_r3,fucci_dmso_after_r3)
dmso_after['Genotype'] = 'DMSO'
dmso_after['Region'] = 'DMSO R3'
dmso_after['Time'] = 1
dmso = pd.concat((dmso,dmso_before,dmso_after),ignore_index=True)

df = pd.concat((tam,dmso),ignore_index=True)

#%%

plt.figure()

pts = plt.scatter(df['H2b'],df['area'],alpha=0.01)

selector = SelectFromCollection(plt.gca(), pts)

#%% Gate the cells

verts = np.array(selector.poly.verts)
x = verts[:,0]
y = verts[:,1]

p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(df['area'],df['H2b'])])
# I = p_.contains_points( np.array([df['area'],df['Corrected Z']]).T )

df_real = df[I]

#%%%

sb.catplot(df_real,x='Region',y='area',hue='Time',kind='violin')

CV = df_real.groupby(['Genotype','Time','Region'])['area'].std()/df_real.groupby(['Genotype','Time','Region'])['area'].mean()
CV = pd.DataFrame(CV)
CV = CV.rename(columns={'area':'CV'})

sb.relplot(CV,x='Time',y='CV',hue='Genotype',kind='line')

#%%
CV = df_real.groupby(['Genotype','Time','Region'])['area'].std()/df_real.groupby(['Genotype','Region','Time'])['area'].mean()
CV = pd.DataFrame(CV)
CV = CV.rename(columns={'area':'CV'})
CV = CV.reset_index()

sb.pointplot(CV,x='Time',y='CV',hue='Region',dodge=True)
CV.to_excel(path.join(dirname,'CV_before_after.xlsx'))

#%% T-test for the fold change in CV

CV = df_real.groupby(['Genotype','Time','Region'])['area'].std()/df_real.groupby(['Genotype','Region','Time'])['area'].mean()

tam_ratio = CV[('4OHT',1)] - CV[('4OHT',0)]
dmso_ratio = CV[('DMSO',1)] - CV[('DMSO',0)]

from scipy.stats import ttest_ind
ttest_ind(tam_ratio.values, dmso_ratio.values)

