#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 18:06:43 2023

@author: xies
"""

from skimage import measure, io
from tqdm import tqdm
from glob import glob
from os import path
import pandas as pd
import seaborn as sb

from matplotlib.path import Path
from SelectFromCollection import SelectFromCollection

#%%

dirname = '/Users/xies/OneDrive - Stanford/Skin/Confocal/02-11-2023 Rb Cre-plusminus Tamoxifen control/H2B Cerulean FUCCI2 Phall-647'
genotypes = ['RBKO','WT']

h2b_filenames = glob(path.join(dirname,'*/*h2b.tif'))
fucci_filenames = glob(path.join(dirname,'*/*fucci.tif'))
mem_filenames = glob(path.join(dirname,'*/*phall.tif'))

seg_filenames = glob(path.join(dirname,'*/cellpose_pruned/*manual.tiff'))

assert(len(h2b_filenames) == len(fucci_filenames))
assert(len(seg_filenames) == len(fucci_filenames))

#%%

_tmp = []
filenames = zip(h2b_filenames,fucci_filenames,seg_filenames)
for i,(h2b_,fucci_,seg_) in enumerate(filenames):
    h2b = io.imread(h2b_)
    fucci = io.imread(fucci_)
    seg = io.imread(seg_)
    
    _df = pd.DataFrame(measure.regionprops_table(seg,intensity_image = h2b,
                                   properties = ['label','area','intensity_mean']))
    df = _df.rename(columns={'area':'Volume','intensity_mean':'H2b mean'})
    
    _df = pd.DataFrame(measure.regionprops_table(seg,intensity_image = fucci,
                                   properties = ['label','intensity_mean']))
    _df = _df.rename(columns={'intensity_mean':'FUCCI mean'})
    
    df = df.merge(_df)
    df['Genotype'] = genotypes[i]
    _tmp.append(df)
    
df = pd.concat(_tmp,ignore_index=True)

#%% Manually gate

# ts = ax.scatter(grid_x, grid_y)

df = pd.concat(_tmp)
# df = pd.concat([wt_table,rbko_table])

plt.figure()

pts = plt.scatter(df['Volume'],df['FUCCI mean'],alpha=0.5)
plt.ylabel('FUCCI intensity')
plt.xlabel('Cell size (fL)')
# plt.xlim([0,25000])
# gate = roipoly()

selector = SelectFromCollection(plt.gca(), pts)

#%% Gate the cells

verts = np.array(selector.poly.verts)
x = verts[:,0]
y = verts[:,1]

p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(df['Volume'],df['FUCCI mean'])])
# I = p_.contains_points( np.array([df['area'],df['Corrected Z']]).T )

g1 = df[I]



