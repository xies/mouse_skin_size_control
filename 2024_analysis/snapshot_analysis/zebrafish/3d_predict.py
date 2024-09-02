from __future__ import print_function, unicode_literals, absolute_import, division
import sys

from glob import glob
from tifffile import imread
from os import path
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap
from stardist.models import StarDist3D

lbl_cmap = random_label_cmap()

#dirname = '/home/xies/data/zebrafish_ditalia/osx_fucci_26hpp_11_4_17/stardist'
dirname='/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/zebrafish_ditalia/osx_fucci_26hpp_11_4_17/stardist/'

files = glob(path.join(dirname,'test_images/*.tif'))
axis_norm = (0,1,2)

n_channel = 1

model = StarDist3D(None,name='stardist',basedir=path.join(dirname,'models'))

for f in files:
	img = imread(f)
	print(f'Predicting {f}')
	img = normalize(img, 1,99.8, axis=axis_norm)
	labels, details = model.predict_instances(img)
	save_tiff_imagej_compatible(path.splitext(f)[0]+'_labels.tif', labels, axes='ZYX')

print('Done with prediction')
