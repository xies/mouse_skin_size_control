import numpy as np
import pandas as pd
from dataclasses import dataclass
from os import path
from skimage import io
import pickle as pkl

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'

dx = 0.26
dz = 2
ymax = 105.04
xmax = 102.96000000000001
zmax = 84

#%% Colorize neighborhood
@dataclass
class Surface:
    vertices: np.array
    faces: np.array
    values: np.array

def load_surface_from_npz(filename,transpose=False):
    arr = np.load(filename)
    vertices = arr['arr_0']
    faces = arr['arr_1']
    values = arr['arr_2']
    if transpose:
        # VTK is in YXZ
        vertices = vertices[:,[2,1,0]]
    surf = Surface(vertices,faces,values)
    return surf

def extract_specific_value_from_multimesh(surf,value2keep):

    vertices = surf.vertices
    faces = surf.faces
    values = surf.values

    I = values == value2keep

    new_values = values[I]
    new_vertices = vertices[I]

    idx_to_keep = np.where(I)[0]
    new_faces = faces[np.all(np.isin(faces, idx_to_keep),axis=1),:]
    new_faces -= new_faces.min() # reset index

    return Surface(new_vertices,new_faces,new_values)

def load_surface_from_pv(mesh,value=None,transpose=False):
    vertices = np.asarray(mesh.points)
    faces = np.asarray(mesh.faces).reshape((-1,4))[:,1:]
    values = np.ones(len(vertices))

    if value is not None:
        if not hasattr(value, "__len__"):
            values = values * value
            print(values)
        else:
            values = value
    if transpose:
        vertices = vertices[:,[2,1,0]]
    surf = Surface(vertices,faces,values)
    return surf

t = 3

im = io.imread(path.join(dirname,f'Channel0-Deconv/Channel0-T{t+1:04d}.tif'))
raw_image = viewer.add_image(im,name='image', scale=[2,.26,.26],rendering='attenuated_mip',blending='additive'
    ,contrast_limits=[0,30000],attenuation=1)

trackID = 1
df = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_organoid_features_dynamic.csv'),index_col=0)
collated = {k:v for k,v in df.groupby('trackID')}
cellID = collated[trackID][collated[trackID].Frame == t]['cellID'].iloc[0]

with open(path.join(dirname,f'geometric_neighbors/geometric_neighbors_T{t+1:04d}.pkl'),'rb') as f:
    cell_neighbors = pkl.load(f)

neighborIDs = cell_neighbors[cellID].values

filename = path.join(dirname,f'manual_seg_mesh/pretty_mesh_T{t+1:04d}.npz')
all_segs = load_surface_from_npz(filename,transpose=False)

central = extract_specific_value_from_multimesh(all_segs, cellID)
central_seg = viewer.add_surface((central.vertices,central.faces,central.values)
         ,name=f'central_{neighborID}',colormap='red')

for neighborID in neighborIDs:
    neighbor_surface = extract_specific_value_from_multimesh(all_segs, neighborID)
    viewer.add_surface((neighbor_surface.vertices,neighbor_surface.faces,neighbor_surface.values)
        ,name=f'neighbor_{neighborID}',colormap='yellow')
