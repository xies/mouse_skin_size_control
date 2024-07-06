import numpy as np
from scipy import ndimage as ndi
from skimage import data
from skimage.morphology import binary_dilation, binary_erosion

import napari

np.random.seed(1)
viewer = napari.Viewer()
blobs = data.binary_blobs(length=128, volume_fraction=0.1, n_dim=2)
image_layer = viewer.add_image(blobs, name='blobs')

# Handle click or drag events separately
@image_layer.mouse_double_click_callbacks.append
def on_second_click_of_double_click(layer, event):
    if layer.name + '_ref_point' in [l.name for l in viewer.layers]:
        viewer.layers[layer.name + '_ref_point'].data = event.position

    else:
        viewer.add_points(event.position, name = layer.name + '_ref_point')

if __name__ == '__main__':
    napari.run()
