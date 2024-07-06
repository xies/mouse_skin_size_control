import napari
import numpy as np

from skimage import io
from os import path
from natsort import natsorted

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/W-R1/3d_nuc_seg/cellpose_cleaned_manual'

filelist = natsort.natsorted(glob(path.join(dirname,'t*.tif')))
segmentation = np.array(list(map(io.imread,filelist)))

# create btrack objects (with properties) from the segmentation data
# (you can also calculate properties, based on scikit-image regionprops)
objects = btrack.utils.segmentation_to_objects(segmentation)

# initialise a tracker session using a context manager
with btrack.BayesianTracker() as tracker:
    # configure the tracker using a config file
    tracker.configure(path.join(dirname,'cell_config.json'))
    # append the objects to be tracked
    tracker.append(objects)

    # set the volume (Z axis volume limits default to [-1e5, 1e5] for 2D data)
#    tracker.volume = ((0, 1200), (0, 1600))

    # track them (in interactive mode)
    tracker.track_interactive(step_size=100)

    # generate hypotheses and run the global optimizer
    tracker.optimize()

    # store the data in an HDF5 file
    tracker.export('/Users/xies/Desktop/tracks.h5', obj_type='obj_type_1')

    # get the tracks as a python list
    tracks = tracker.tracks

    # optional: get the data in a format for napari
    data, properties, graph = tracker.to_napari()
