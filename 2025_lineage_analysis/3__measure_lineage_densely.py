#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 16:48:47 2025

@author: xies
"""

# Core libraries
import numpy as np
from skimage import io, util, morphology, exposure, measure
from scipy.spatial import Voronoi, Delaunay
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb

# Specific utils
from imageUtils import draw_labels_on_image, colorize_segmentation, normalize_exposure_by_axis
from trimesh import Trimesh, geometry
from trimesh.curvature import discrete_gaussian_curvature_measure, \
    discrete_mean_curvature_measure, sphere_ball_intersection
import pyvista as pv

# General utils
from tqdm import tqdm
from os import path,makedirs
import pickle as pkl

dt = 12 #hrs
dx = 0.25
dz = 1
Z_SHIFT = 10
KAPPA = 5 # microns

# for expansion
footprint = morphology.cube(3)

# Filenames
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

#%%

from measurements import measure_nuclear_geometry_from_regionprops, \
        measure_cyto_geometry_from_regionprops, measure_cyto_intensity, measure_flat_cyto_from_regionprops, \
        estimate_sh_coefficients, find_distance_to_closest_point, \
        measure_collagen_structure, get_mesh_from_bm_image, get_tissue_curvatures, export_mesh

SAVE = True
VISUALIZE = False
LMAX = 5 # Number of spherical harmonics components to calculate

all_df = []

# Load manual tracking
with open(path.join(dirname,'Mastodon/dense_tracks.pkl'),'rb') as file:
    tracks = pkl.load(file)
manual = pd.concat(tracks,ignore_index=True)
manual['Frame'] = manual['Frame'].astype(float).astype(int)
manual = manual.set_index(['Frame','TrackID'])
manual = manual.drop(columns=['ID','X','Y','Z','Border'])

# Load segmentations
tracked_nuc = io.imread(path.join(dirname,'Mastodon/tracked_nuc.tif'))
tracked_cyto = io.imread(path.join(dirname,'Mastodon/tracked_cyto.tif'))
tracked_manual_cyto = io.imread(path.join(dirname,'Mastodon/tracked_manual_cyto.tif'))

# Load channels
h2b = io.imread(path.join(dirname,'Cropped_images/B.tif'))
h2b = normalize_exposure_by_axis(h2b,axis=0)
fucci_g1 = io.imread(path.join(dirname,'Cropped_images/R.tif'))
fucci_g1 = normalize_exposure_by_axis(fucci_g1,axis=0)

for t in tqdm(range(15)):

    #----- read segmentation files -----
    nuc_seg = tracked_nuc[t,...]
    cyto_seg = tracked_cyto[t,...]
    ZZ,YY,XX = nuc_seg.shape

    # --- 1. Voxel-based cell geometry measurements ---
    df_nuc = measure_nuclear_geometry_from_regionprops(nuc_seg,spacing = [dz,dx,dx])
    df_cyto = measure_cyto_geometry_from_regionprops(cyto_seg,spacing = [dz,dx,dx])
    df_manual = pd.DataFrame(measure.regionprops_table(tracked_manual_cyto[t,...],
                                          properties=['label','area']))
    df_manual = df_manual.rename(columns={'label':'TrackID',
                                          'area':'Manual cell volume'})
    df = pd.merge(left=df_nuc,right=df_cyto,left_on='TrackID',right_on='TrackID',how='left')
    df = pd.merge(left=df,right=df_manual,left_on='TrackID',right_on='TrackID',how='left')
    df['Frame'] = t
    df['Time'] = t * dt
    int_images = {'H2B':h2b[t,...],'FUCCI':fucci_g1[t,...]}
    intensity_df = measure_cyto_intensity(cyto_seg,int_images)
    df = pd.merge(left=df,right=intensity_df,left_on='TrackID',right_on='TrackID',how='left')

    # ----- 3. Use flattened 3d cortical segmentation and measure geometry and collagen
    # from cell-centric coordinates ----
    f = path.join(dirname,f'Image flattening/flat_tracked_cyto/t{t}.tif')
    flat_cyto = io.imread(f)

    # Calculate collagen structuring matrix
    collagen_image = io.imread(path.join(dirname,f'Image flattening/flat_collagen/t{t}.tif'))
    (Jxx,Jxy,Jyy) = measure_collagen_structure(collagen_image,blur_sigma=3)

    df_flat,basal_masks_2save = measure_flat_cyto_from_regionprops(
        flat_cyto, collagen_image, (Jxx, Jyy, Jxy), spacing = [dz,dx,dx])
    df = pd.merge(df,df_flat,left_on='TrackID',right_on='TrackID',how='left')

    if not path.exists(path.join(dirname,'Image flattening/basal_masks')):
        makedirs(path.join(dirname,'Image flattening/basal_masks'))
    if SAVE:
        io.imsave(path.join(dirname,f'Image flattening/basal_masks/t{t}.tif'),basal_masks_2save)

    # Book-keeping
    df = df.drop(columns=['bbox-1','bbox-2','bbox-4','bbox-5'])
    df['X-pixels'] = df['X'] / dx
    df['Y-pixels'] = df['Y'] / dx

    # Derive NC ratio
    df['NC ratio'] = df['Nuclear volume'] / df['Cell volume']

    dense_coords = np.array([df['Y-pixels'],df['X-pixels']]).T
    dense_coords_3d_um = np.array([df['Z'],df['Y'],df['X']]).T

    #----- 4. Nuc-to-BM heights -----
    # Load heightmap and calculate adjusted height
    heightmap = io.imread(path.join(dirname,f'Image flattening/heightmaps/t{t}.tif'))
    heightmap_shifted = heightmap + Z_SHIFT
    df['BM height'] = heightmap_shifted[
        np.round(df['Y']).astype(int),np.round(df['X']).astype(int)]
    df['Height to BM'] = df['BM height'] - df['Z'].values
    # stop

    #----- Find border cells -----
    # Generate a dense mesh based sole only 2D/3D nuclear locations
    #% Use Delaunay triangulation in 2D to approximate the basal layer topology
    tri_dense = Delaunay(dense_coords)
    # Use the dual Voronoi to get rid of the border/infinity cells
    vor = Voronoi(dense_coords)
    # Find the voronoi regions that touch the image border
    vertices_outside = np.where(np.any((vor.vertices < 0) | (vor.vertices > XX),axis=1))
    regions_outside = np.where([ np.any(np.in1d(np.array(r), vertices_outside)) for r in vor.regions])
    regions_outside = np.hstack([regions_outside, np.where([-1 in r for r in vor.regions])])
    Iborder = np.in1d(vor.point_region, regions_outside)
    border_nuclei = df.loc[Iborder].index
    df['Border'] = False
    df.loc[ border_nuclei, 'Border'] = True

    #----- Cell coordinates mesh for geometry -----
    # Generate 3D mesh for curvature analysis -- no need to specify precise cell-cell junctions
    Z,Y,X = dense_coords_3d_um.T
    cell_coords_mesh = Trimesh(vertices = np.array([X,Y,Z]).T, faces=tri_dense.simplices)
    mean_curve_coords = -discrete_mean_curvature_measure(
        cell_coords_mesh, cell_coords_mesh.vertices, 5)/sphere_ball_intersection(1, 5)
    gaussian_curve_coords = -discrete_gaussian_curvature_measure(
        cell_coords_mesh, cell_coords_mesh.vertices, 5)/sphere_ball_intersection(1, 5)
    df['Mean curvature - cell coords'] = mean_curve_coords
    df['Gaussian curvature - cell coords'] = gaussian_curve_coords

    # ---- 5. Get 3D mesh from the BM image ---
    bm_height_image = io.imread(path.join(dirname,f'Image flattening/height_image/t{t}.tif'))
    bg_mesh = get_mesh_from_bm_image(bm_height_image,spacing=[dz,dx,dx],decimation_factor=30)
    
    # Export mesh: vert, face, value (normals) for napari usage
    export_mesh(bg_mesh,path.join(dirname,f'Image flattening/trimesh/t{t}.npz'))
    # vertices = np.asarray(bg_mesh.vertices[:,[2,1,0]])
    # faces = np.asarray(bg_mesh.faces)
    # normals = geometry.mean_vertex_normals(len(bg_mesh.vertices),bg_mesh.faces,bg_mesh.face_normals)
    # values = np.dot(normals,[1,-1,1])
    # np.savez(path.join(dirname,f'Image flattening/trimesh/t{t}.npz'),
    #           vertices = vertices,faces = faces,values = values)
    
    closest_mesh_to_cell,_,_ = bg_mesh.nearest.on_surface(dense_coords_3d_um[:,::-1])
    for kappa in [2,5,10,15]:
        mean_curve, gaussian_curve = get_tissue_curvatures(bg_mesh,
                                                           kappa=kappa, query_pts = closest_mesh_to_cell)
    
        df[f'Mean curvature {kappa}um'] = mean_curve
        df[f'Gaussian curvature {kappa}um'] = gaussian_curve
    

    # --- 6. 3D shape decomposition ---

    # 2a: Estimate cell and nuclear mesh using spherical harmonics
    sh_coefficients = estimate_sh_coefficients(c, n, LMAX, spacing = [dz,dx,dx])
    sh_coefficients = sh_coefficients.set_index('TrackID')
    # sh_coefficients.columns = 'cyto_' + sh_coefficients.columns
    df = pd.merge(df,sh_coefficients,left_on='TrackID',right_on='TrackID',how='left')
    # sh_coefficients = estimate_sh_coefficients(nuc_seg, LMAX, spacing = [dz,dx,dx])
    # sh_coefficients = sh_coefficients.set_index('TrackID')
    # sh_coefficients.columns = 'nuc_' + sh_coefficients.columns
    # df = pd.merge(df,sh_coefficients,left_on='TrackID',right_on='TrackID',how='left')

    #----- Use macrophage annotations to find distance to them -----
    #NB: the macrophage coords are in um
    macrophage_xyz = pd.read_csv(path.join(dirname,f'3d_cyto_seg/macrophages/t{t}.csv'))
    macrophage_xyz = macrophage_xyz.rename(columns={'axis-0':'Z','axis-1':'Y','axis-2':'X'})
    macrophage_xyz['X'] = macrophage_xyz['X'] * dx
    macrophage_xyz['Y'] = macrophage_xyz['Y'] * dx
    df['Distance to closest macrophage'] = find_distance_to_closest_point(pd.DataFrame(dense_coords_3d_um,columns=['Z','Y','X']), macrophage_xyz)

    # Load basal masks for current frame
    frame_basal_mask = io.imread(path.join(dirname,f'Image flattening/basal_masks/t{t}.tif'))

    # Merge with manual annotations
    df = df.reset_index(drop=True)
    # df = df.set_index(['Frame','TrackID'])

    # Save the DF
    all_df.append(df)

    #Save a bunch of intermediates
    # Save segmentation with text labels @ centroid
    # im_cellposeID = draw_labels_on_image(dense_coords,df_dense['basalID'],[XX,XX],font_size=12)
    # im_cellposeID.save(path.join(dirname,f'3d_nuc_seg/cellposeIDs/t{t}.tif'))

all_df = pd.concat(all_df,ignore_index=False)
all_df = all_df.set_index(['Frame','TrackID'])
all_df = pd.merge(all_df,manual,left_on=['Frame','TrackID'],right_on=['Frame','TrackID'],how='left')

# Sanitize field dtypes that are NaN from the merge with manual
all_df['Cell type'] = all_df['Cell type'].astype(str)
all_df['Terminus'] = (all_df['Terminus'].astype(float) == 1)
all_df['Cutoff'] = (all_df['Cutoff'].astype(float) == 1)
all_df['Reviewed'] = (all_df['Reviewed'].astype(float) == 1)
all_df['Complete cycle'] = (all_df['Complete cycle'].astype(float) == 1)

all_df.to_csv(path.join(dirname,'Mastodon/single_timepoints.csv'))

#%% PCA diagonalize the shcoeffs across both regions, remove original features, and put the shcoeff_PCAs

from sklearn import decomposition

dirnames = {'R1':'/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/',
            'R2':'/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'}

regions = {}
for name,dirname in dirnames.items():
    regions[name] = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints.csv'))
    regions[name]['Region'] = name

df_concat = pd.concat(regions.values(),ignore_index=True)

# Grab all nuc_coeffs
nuc_coef_cols = [f for f in df_concat.columns if 'nuc_shcoeff' in f and 'surface_area' not in f]
# Grab all cyto_coeffs
cyto_coef_cols = [f for f in df_concat.columns if 'cyto_shcoeff' in f and 'surface_area' not in f]

Inonans = df_concat[cyto_coef_cols].dropna(axis=0).index
pca = decomposition.PCA()
PCA = pca.fit_transform(df_concat.loc[Inonans,nuc_coef_cols+cyto_coef_cols])
component_cutoff = 9

# Put back the PCA coeffients region by region
PCA = pd.DataFrame(PCA[:,:component_cutoff],
                       columns = [f'nuc_shcoeff_PC{i}' for i in range(component_cutoff)])
PCA[['Region','Frame','TrackID']] = df_concat.loc[Inonans][['Region','Frame','TrackID']].values

for name,region in regions.items():
    # Prepare indexes
    this_PCA = PCA[PCA['Region'] == name]
    this_PCA = this_PCA.drop(columns='Region').set_index(['Frame','TrackID']).astype(float)
    
    # Merge
    region = region.set_index(['Frame','TrackID'])
    region_pc = pd.merge(region,this_PCA, right_index=True, left_index=True, how='left')
    
    # Drop old cofficients
    region_pc = region_pc.drop(columns=nuc_coef_cols+cyto_coef_cols)
    region_pc = region_pc.drop(columns='Region')
    # Save
    region_pc.to_csv(path.join(dirnames[name],'Mastodon/single_timepoints_pca.csv'))


#%% Find missing cyto segs

non_border_basals = all_df[(all_df['Cell type'] =='Basal') & (~all_df['Border'])]
non_border_basals.index[non_border_basals['Cell volume'].isnull()].tolist()
missing_cytos = non_border_basals.index[non_border_basals['Cell volume'].isnull()].tolist()

print(missing_cytos[:50])

#%% Re-export all the BM mesh objects into individual a single unified TZYX mesh for display in napari

mesh_tuples = [np.load(path.join(dirname,f'Image flattening/trimesh/t{t}.npz')) for t in range(15)]

# Vertices -> append a T dimention
vertices_raw = [m['vertices'] for m in mesh_tuples]
num_vert_per_frame = [len(v) for v in vertices_raw]
num_verts_so_far = np.cumsum(num_vert_per_frame)
vertices = vertices_raw.copy()
for t,v in enumerate(vertices_raw):
    num_verts = len(v)
    vertices[t] = np.hstack((np.ones((num_verts,1)) * t, v))
vertices = np.vstack(vertices)

# Faces: for every t > 0, calculate the cumulative # of vertices so far, and increase
# vertex index by that amount
faces_raw = [m['faces'] for m in mesh_tuples]
faces = faces_raw.copy()
for t,f in enumerate(faces_raw):
    if t > 0:
        faces[t] = f + num_verts_so_far[t-1]
faces = np.vstack(faces)

values_raw = [m['values'].T for m in mesh_tuples]
values = np.hstack(values_raw)

np.savez(path.join(dirname,'Image flattening/trimesh/bg_surface_timeseries.npz'),
         vertices=vertices,faces=faces,values=values)

#%% Export lineage-colorized tracks

from imageUtils import colorize_segmentation
import tifffile

lineage_tree = {trackID:t.iloc[0]['LineageID'] for trackID,t in all_df.groupby('TrackID')}
lineage_image = colorize_segmentation(tracked_cyto,lineage_tree)

tifffile.imwrite(path.join(dirname,'Mastodon/lineageID_cyto.tif'),
                 lineage_image, metadata={'axes': 'TZYX'}, compression ='zlib')

lineage_tree = {trackID:t.iloc[0]['LineageID'] for trackID,t in all_df.groupby('TrackID')}
lineage_image = colorize_segmentation(tracked_nuc,lineage_tree)

tifffile.imwrite(path.join(dirname,'Mastodon/lineageID_nuc.tif'),
                 lineage_image, metadata={'axes': 'TZYX'}, compression ='zlib')


#%% Dimensionality reduce the SPH coefficients
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# all_df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints.csv'),index_col=['Frame','TrackID'])
# df = all_df

# # Nuclear first
# nuc_columns = df.columns[df.columns.str.startswith('nuc_')]
# # nuc_columns = nuc_columns[df[nuc_columns].sum() > 0]
# # Drop NaN columns (these are often late mitotic figures)
# Ivalid = np.any(df[nuc_columns].notna(),axis=1) & ~df.Border
# scaler = StandardScaler()
# nuc_sph = np.array( scaler.fit_transform(df.loc[Ivalid,nuc_columns]) )
# pca = PCA()
# nuc_sph_PCA = pca.fit_transform(nuc_sph)

# loading_matrix = pd.DataFrame(pca.components_, columns=nuc_columns)

#%% Visualize the different PCA dimensions

# pl = pv.Plotter()

# for comp in range(0,10):

#     new_dict = dict(zip(loading_matrix.columns.str.split('nuc_',expand=True).get_level_values(1),loading_matrix.loc[comp]))
#     M = shtools.convert_coeffs_dict_to_matrix(new_dict,lmax=10)
#     mesh,_ = shtools.get_even_reconstruction_from_coeffs(M,npoints=1024)


#     pl.add_mesh(pv.wrap(mesh), opacity=0.5)

# pl.show_axes()
# pl.show()

# pl.save_graphic(path.join(dirname,f'shape_mode_analysis/nuc_modes/comp_{comp}.pdf'))


#%% Colorize cell with measurement for visualization

# t = 0

# measurement = all_df.loc[t,:]['Mean curvature - cell coords']
# measurement /= np.max([measurement.max(),np.abs(measurement.min())])
# colorized = colorize_segmentation(tracked_nuc[t,...],
#                       measurement.to_dict(),dtype=float)
# io.imsave('/Users/xies/Desktop/cell_coords.tif',
#           util.img_as_int(exposure.rescale_intensity(colorized,in_range=(-1,1)) ) )


# measurement = all_df.loc[t,:]['Mean curvature']
# measurement /= np.max([measurement.max(),np.abs(measurement.min())])
# colorized = colorize_segmentation(tracked_nuc[t,...],
#                       measurement.to_dict(),dtype=float)
# io.imsave('/Users/xies/Desktop/bm.tif',
#           util.img_as_int(exposure.rescale_intensity(colorized,in_range=(-1,1)) ) )


# measurement = all_df.loc[t,:]['Collagen intensity']
# measurement /= np.max([measurement.max(),np.abs(measurement.min())])
# colorized = colorize_segmentation(tracked_nuc[t,...],
#                       measurement.to_dict(),dtype=float)
# io.imsave('/Users/xies/Desktop/collagen_intensity.tif',
#           util.img_as_int(exposure.rescale_intensity(colorized,in_range=(-1,1)) ) )


# measurement = all_df.loc[t,:]['Collagen coherence']
# measurement /= np.max([measurement.max(),np.abs(measurement.min())])
# colorized = colorize_segmentation(tracked_nuc[t,...],
#                       measurement.to_dict(),dtype=float)
# io.imsave('/Users/xies/Desktop/collagen_coherence.tif',
#           util.img_as_int(exposure.rescale_intensity(colorized,in_range=(-1,1)) ) )
