import os,sys
import tensorflow as tf

import voxelmorph as vxm
import neurite as ne
import numpy as np

# download MRI tutorial data
# !wget https://surfer.nmr.mgh.harvard.edu/pub/data/voxelmorph/tutorial_data.tar.gz -O data.tar.gz
# !tar -xzvf data.tar.gz
import matplotlib.pyplot as plt

def vxm_data_generator_3d(x_xata,batch_size=32):
    '''
    x-data: array of NxZxYxX where N is # of samples, ZYX are dims
    '''
    vol_shape = x_data.shape[1:]
    ndims = len(vol_shape)
    assert(ndims == 3)

    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    #i.e. [32, Z, Y, X, 3]

    while True:
        # inputs: size [32, Z,Y,X, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size) #randomly subset batchsize for moving
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_iamges, fixed_images]

        # outputs: true moved image (don't have access, so try to )



# For now
# our data will be of shape 160 x 192 x 224
vol_shape = (160,1024,1024)
nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16]
]
# build vxm network
vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0);

moving = np.load('/Users/xies/Desktop/test/t14.npy')
fixed = np.load('/Users/xies/Desktop/test/t13.npy')

val_input = [
    moving[np.newaxis, ... ,np.newaxis],
    fixed[np.newaxis, ... ,np.newaxis]
]

vxm_model.load_weights('brain_3d.h5')
val_pred = vxm_model.predict(val_input);

moved_pred = val_pred[0].squeeze()
pred_warp = val_pred[1]

#Visualize
moved_pred.shape

mid_slices_fixed = [np.take(moving, vol_shape[d]//2, axis=d) for d in range(3)]
mid_slices_fixed[1] = np.rot90(mid_slices_fixed[1], 1)
mid_slices_fixed[2] = np.rot90(mid_slices_fixed[2], -1)

mid_slices_pred = [np.take(moved_pred, vol_shape[d]//2, axis=d) for d in range(3)]
mid_slices_pred[1] = np.rot90(mid_slices_pred[1], 1)
mid_slices_pred[2] = np.rot90(mid_slices_pred[2], -1)
ne.plot.slices(mid_slices_fixed + mid_slices_pred, cmaps=['gray'], do_colorbars=True, grid=[2,3]);
