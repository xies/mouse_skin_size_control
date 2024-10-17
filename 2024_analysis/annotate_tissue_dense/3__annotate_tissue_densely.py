#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 00:02:34 2022

@author: xies
"""

import numpy as np
from skimage import io, measure, draw, util, morphology
from scipy.spatial import distance, Voronoi, Delaunay
import pandas as pd

from trimesh import Trimesh
from trimesh.curvature import discrete_gaussian_curvature_measure, discrete_mean_curvature_measure, sphere_ball_intersection
from basicUtils import nonans

import matplotlib.pylab as plt
from imageUtils import draw_labels_on_image, draw_adjmat_on_image, most_likely_label, colorize_segmentation
from mathUtils import get_neighbor_idx, surface_area, parse_3D_inertial_tensor, argsort_counter_clockwise

from tqdm import tqdm
from os import path,makedirs

dx = 0.25
Z_SHIFT = 10

# Differentiating thresholds
centroid_height_cutoff = 3.5 #microns above BM

# Filenames
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

# Demo files
# dirname = '/Users/xies/Desktop/Code/mouse_skin_size_control/2024_analysis/test_dataset/'
# im_demo = io.imread(path.join(dirname,'example_mouse_skin_image.tif'))
# flattened_3d_seg_demo = io.imread(path.join(dirname,'flattened_segmentation.tif'))
# collagen_gradients_demo = io.imread(path.join(dirname,'collagen_gradients.tif'))
# heightmaps_demo = io.imread(path.join(dirname,'heightmaps.tif'))

# FUCCI threshold (in stds)
alpha_threshold = 1
dx = 0.25
dz = 1
#NB: idx - the order in array in dense segmentation

#%% Load the segmentation and coordinates

def find_differentiating_cells(df,height_cutoff,heightmap):
    
    # I = df['Height to BM'] > height_cutoff
    diff_height_th = heightmap[np.round(
        df['Y-pixels']).astype(int),np.round(df['X-pixels']).astype(int)] - height_cutoff
    
    # Check that the bottom bbox of nucleus is not within cutoff of the 'heightmap'
    I = df['Nuclear bbox bottom'] < diff_height_th
    
    df['Differentiating'] = False
    df.loc[I,'Differentiating'] = True
    
    return df

# convert trianglulation to adjacency matrix (for easy editing)
def tri_to_adjmat(tri):
    num_verts = max(map(max,tri.simplices)) + 1
    A = np.zeros((num_verts,num_verts),dtype=bool)
    for idx in range(num_verts):
        neighbor_idx = get_neighbor_idx(tri,idx)
        A[idx,neighbor_idx] = True
    return A

# Find distance to nearest manually annotated point in points-list
def find_distance_to_closest_point(dense_3d_coords,annotation_coords_3d):
    distances = np.zeros(len(dense_3d_coords))

    for i,row in dense_3d_coords.iterrows():
        dx = row['X'] - annotation_coords_3d['X']
        dy = row['Y'] - annotation_coords_3d['Y']
        dz = row['Z'] - annotation_coords_3d['Z']
        D = np.sqrt(dx**2 + dy**2 + dz**2)
        
        distances[i] = D.min()
            
    return distances
        
#%%

SAVE = True
VISUALIZE = False

DEMO = False

df = []

# for expansion
footprint = morphology.cube(3)

for t in tqdm(range(15)):
    
    #----- cell-centric msmts -----
    if DEMO:
        nuc_dense_seg = im_demo[t,:,3,:,:]
        cyto_dense_seg = im_demo[t,:,4,:,:]
        manual_tracks = im_demo[t,:,5,:,:]
    else:
        nuc_dense_seg = io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}.tif'))
        cyto_dense_seg = io.imread(path.join(dirname,f'3d_cyto_seg/3d_cyto_manual/t{t}_cleaned.tif'))
        manual_tracks = io.imread(path.join(dirname,f'manual_basal_tracking/sequence/t{t}.tif'))
    
    _,YY,XX = nuc_dense_seg.shape
    # Initialize measurements from nuclear segmentation
    df_dense = pd.DataFrame( measure.regionprops_table(nuc_dense_seg, intensity_image=cyto_dense_seg
                                                       ,properties=['label','area','centroid','solidity','bbox']
                                                       ,extra_properties = [most_likely_label]
                                                       ,spacing = [dz,dx,dx]))
    df_dense = df_dense.rename(columns={'centroid-0':'Z','centroid-1':'Y','centroid-2':'X'
                                        ,'label':'CellposeID','area':'Nuclear volume'
                                        ,'bbox-0':'Nuclear bbox top'
                                        ,'bbox-3':'Nuclear bbox bottom'
                                        ,'solidity':'Nuclear solidity'
                                        ,'most_likely_label':'CytoID'})
    df_dense.loc[df_dense['CytoID'] == 0,'CytoID'] = np.nan
    
    # Measurements from cortical segmentation
    df_cyto = pd.DataFrame( measure.regionprops_table(cyto_dense_seg,intensity_image=nuc_dense_seg
                                                      , properties=['label','area','centroid']
                                                      , extra_properties=[most_likely_label]
                                                      , spacing=[dz,dx,dx]))
    df_cyto = df_cyto.rename(columns={'area':'Cell volume','label':'CytoID','most_likely_label':'CellposeID'
                                      ,'centroid-0':'Z-cell','centroid-1':'Y-cell','centroid-2':'X-cell'})
    # Initiate fields
    df_cyto['Axial component'] = np.nan
    df_cyto['Planar component '] = np.nan
    df_cyto['Planar component 2'] = np.nan
    df_cyto['Axial angle'] = np.nan
    df_cyto['Planar angle'] = np.nan
    df_cyto['Apical area'] = np.nan
    df_cyto['Middle area'] = np.nan
    df_cyto['Basal area'] = np.nan
    df_cyto['Basal orientation'] = np.nan
    df_cyto['Basal eccentricity'] = np.nan
    df_cyto['Height'] = np.nan
    df_cyto['Collagen orientation'] = np.nan
    df_cyto['Collagen fibrousness'] = np.nan
    df_cyto['Collagen alignment'] = np.nan
    
    df_cyto.index = df_cyto['CytoID'] # index w CytoID so we can key from regionprops
    df_cyto['Cell volume'] = df_cyto['Cell volume']
    df_cyto['Y-cell'] = df_cyto['Y-cell']
    df_cyto['X-cell'] = df_cyto['X-cell']
    for p in measure.regionprops(cyto_dense_seg, spacing=[dz,dx,dx]):
        i = p['label']
        I = p['inertia_tensor']
        Iaxial, phi, Ia, Ib, theta = parse_3D_inertial_tensor(I)
        df_cyto.at[i,'Axial component'] = Iaxial
        df_cyto.at[i,'Planar component 1'] = Ia
        df_cyto.at[i,'Planar component 2'] = Ib
        df_cyto.at[i,'Axial angle'] = phi
        df_cyto.at[i,'Planar angle'] = theta
    
    # ----- Load flattened 3d cortical segmentation and measure geometry from cell-centric coordinates ----
    if DEMO:
        im = flattened_3d_seg_demo[t,...]
    else:
        f = path.join(dirname,f'Image flattening/flat_3d_cyto_seg/t{t}.tif')
        im = io.imread(f)
    
    # Load the structuring matrix elements for collagen
    if DEMO:
        [Gx,Gy] = collagen_gradients_demo[t,:,:,:]
    else:
        f = path.join(dirname,f'Image flattening/collagen_orientation/t{t}.npy')
        [Gx,Gy] = np.load(f)
    Jxx = Gx*Gx
    Jxy = Gx*Gy
    Jyy = Gy*Gy
    
    properties = measure.regionprops(im, extra_properties = [surface_area]) # Avoid using table bc of bbox
    basal_masks_2save = np.zeros((YY,XX))
    for p in properties:
        
        cytoID = p['label']
        bbox = p['bbox']
        Z_top = bbox[0]
        Z_bottom = bbox[3]
        
        mask = im == cytoID
        apical_area = mask[Z_top:Z_top+3,...].max(axis=0)
        apical_area = apical_area.sum()
        
        # mid-level area
        mid_area = mask[np.round((Z_top+Z_bottom)/2).astype(int),...].sum()
        
        basal_mask = mask[Z_bottom-4:Z_bottom,...]
        basal_mask = basal_mask.max(axis=0)
        basal_area = basal_mask.sum()
    
        basal_masks_2save[basal_mask] = df_cyto.at[cytoID,'CytoID']

        #NB: skimage uses the 'vertical' as the orientation axis
        basal_orientation = measure.regionprops(basal_mask.astype(int))[0]['orientation']
        basal_eccentricity = measure.regionprops(basal_mask.astype(int))[0]['eccentricity']
        # Need to 'convert to horizontal--> subtract 90-deg from image
        basal_orientation = np.rad2deg(basal_orientation + np.pi/2)
        df_cyto.at[cytoID,'Apical area'] = apical_area * dx**2
        df_cyto.at[cytoID,'Basal area'] = basal_area * dx**2
        
        df_cyto.at[cytoID,'Basal orientation'] = basal_orientation
        df_cyto.at[cytoID,'Basal eccentricity'] = basal_eccentricity
        
        # Subtract the mid-area of central cell from the coronal area
        df_cyto.at[cytoID,'Middle area'] = mid_area * dx**2
        df_cyto.at[cytoID,'Height'] = Z_bottom-Z_top
        
        # Characteristic matrix of collagen signal
        J = np.matrix( [[Jxx[basal_mask].sum(),Jxy[basal_mask].sum()],[Jxy[basal_mask].sum(),Jyy[basal_mask].sum()]] )
        l,D = np.linalg.eig(J) # NB: not sorted
        order = np.argsort(l)[::-1] # Ascending order
        l = l[order]
        D = D[:,order]
        
        # Orientation
        theta = np.rad2deg(np.arctan(D[1,0]/D[0,0]))
        fibrousness = (l[0] - l[1]) / l.sum()
        
        # theta = np.rad2deg( -np.arctan( 2*Jxy[basal_mask].sum() / (Jyy[basal_mask].sum()-Jxx[basal_mask].sum()) )/2 )
        # # Fibrousness
        # fib = np.sqrt((Jyy[basal_mask].sum() - Jxx[basal_mask].sum())**2 + 4 * Jxy[basal_mask].sum()) / \
        #     (Jxx[basal_mask].sum() + Jyy[basal_mask].sum())
        df_cyto.at[cytoID,'Collagen orientation'] = theta
        df_cyto.at[cytoID,'Collagen fibrousness'] = fibrousness
        df_cyto.at[cytoID,'Collagen alignment'] = np.abs(np.cos(theta - basal_orientation))
    
    if not path.exists(path.join(dirname,'Image flattening/basal_masks')):
        makedirs(path.join(dirname,'Image flattening/basal_masks'))
    io.imsave(path.join(dirname,f'Image flattening/basal_masks/t{t}.tif'),basal_masks_2save)
    
    # Book-keeping
    df_dense = df_dense.drop(columns=['bbox-1','bbox-2','bbox-4','bbox-5'])
    # df_dense['Nuclear volume'] = df_dense['Nuclear volume']
    df_dense['X-pixels'] = df_dense['X'] / dx
    df_dense['Y-pixels'] = df_dense['Y'] / dx
    df_dense['Frame'] = t
    df_dense['basalID'] = np.nan
    
    # Merge NUC and CYTO annotations into the same dataframe, keyed on NUCLEAR cellposeID, and keep nuclear in merge
    # because df_cyto is generally a subset of df_nuc
    df_dense = df_dense.merge(df_cyto,on='CellposeID',how='left')
    # assert(np.all(np.isnan(df_dense[df_dense['CytoID_x'] != df_dense['CytoID_y']]['CytoID_x'])))
    df_dense = df_dense.drop(columns='CytoID_y')
    df_dense = df_dense.rename(columns={'CytoID_x':'CytoID'})
    # Derive NC ratio
    df_dense['NC ratio'] = df_dense['Nuclear volume'] / df_dense['Cell volume']
   
    #----- Load FUCCI channel + auto annotate cell cycle ---
    if DEMO:
        R = im_demo[t,:,0,:,:]
    else:
        im = io.imread(path.join(dirname,f'im_seq/t{t}.tif'))
        R = im[...,0]
    for i,p in enumerate(measure.regionprops(nuc_dense_seg,intensity_image=R)):
        df_dense.at[i,'FUCCI intensity'] = p['intensity_mean']
        bbox = p['bbox']
        local_fucci = R[bbox[0]:bbox[3],bbox[1]:bbox[4],bbox[2]:bbox[5]]
        local_seg = p['image']
        footprint = morphology.cube(3)
        local_seg = morphology.dilation(local_seg,footprint=footprint)
        
        bg_mean = local_fucci[~local_seg].mean()
        bg_std = local_fucci[~local_seg].std()
        df_dense.at[i,'FUCCI background mean'] = bg_mean
        df_dense.at[i,'FUCCI background std'] = bg_std
        df_dense.at[i,'FUCCI bg sub'] = p['intensity_mean'] - bg_mean
        
    df_dense['FUCCI thresholded'] = 'Low'
    I = df_dense['FUCCI intensity'] > df_dense['FUCCI background mean'] + alpha_threshold * df_dense['FUCCI background std']
    df_dense.loc[I,'FUCCI thresholded'] = 'High'
   
    #----- Various nuclear volume annotations -----
    # Use thresholded mask to calculate nuclear volume
    # Load raw images or pre-made masks
    if not DEMO:
        mask = io.imread(path.join(dirname,f'Misc/H2b masks/t{t}.tif')).astype(bool)
        this_seg_dilated = morphology.dilation(nuc_dense_seg,footprint=footprint)
        this_seg_dilated[~mask] = 0
        threshed_volumes = pd.DataFrame(measure.regionprops_table(
            this_seg_dilated, properties=['label','area'], spacing=[dz,dx,dx] ))
        threshed_volumes['area'] = threshed_volumes['area']
        threshed_volumes = threshed_volumes.rename(columns={'label':'CellposeID','area':'Nuclear volume th'})
        df_dense = df_dense.merge(threshed_volumes,on='CellposeID', how='outer')
    
        # Calculate a 'normalized nuc volume'
        norm_factor = df_dense[df_dense['FUCCI thresholded'] == 'High']['Nuclear volume th'].mean()
        df_dense['Nuclear volume normalized'] = df_dense['Nuclear volume th']/norm_factor


    #----- map from cellpose to manual -----
    #NB: best to use the manual mapping since it guarantees one-to-one mapping from cellpose to manual cellIDs
    df_manual = pd.DataFrame(measure.regionprops_table(manual_tracks,intensity_image = nuc_dense_seg,
                                                       properties = ['label'],
                                                       extra_properties = [most_likely_label]))
    df_manual = df_manual.rename(columns={'label':'basalID','most_likely_label':'CellposeID'})
    # assert(np.isnan(df_manual['CellposeID']).sum() == 0)
    
    # Reverse the mapping from CellposeID to basalID
    for _,this_cell in df_manual.iterrows():
         df_dense.loc[ df_dense['CellposeID'] == this_cell['CellposeID'],'basalID'] = this_cell['basalID']

    dense_coords = np.array([df_dense['Y-pixels'],df_dense['X-pixels']]).T
    dense_coords_3d_um = np.array([df_dense['Z'],df_dense['Y'],df_dense['X']]).T
    
    #----- Nuc-to-BM heights -----
    # Load heightmap and calculate adjusted height
    if DEMO:
        heightmap = heightmaps_demo[t,...]
    else:
        heightmap = io.imread(path.join(dirname,f'Image flattening/heightmaps/t{t}.tif'))
    heightmap_shifted = heightmap + Z_SHIFT
    df_dense['Height to BM'] = heightmap_shifted[np.round(df_dense['Y']).astype(int),np.round(df_dense['X']).astype(int)] - df_dense['Z']
    
    
    # Based on adjusted height, determine a 'cutoff'
    # df_dense['Differentiating'] = df_dense['Height to BM'] > HEIGHT_CUTOFF
    df_dense = find_differentiating_cells(df_dense,centroid_height_cutoff,heightmap)
    
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
    border_nuclei = df_dense.loc[Iborder]['CellposeID'].values
    df_dense['Border'] = False
    df_dense.loc[ np.in1d(df_dense['CellposeID'],border_nuclei), 'Border'] = True
    
    #----- 3D cell mesh for geometry -----
    # Generate 3D mesh for curvature analysis -- no need to specify precise cell-cell junctions
    Z,Y,X = dense_coords_3d_um.T
    mesh = Trimesh(vertices = np.array([X,Y,Z]).T, faces=tri_dense.simplices)
    # mesh_sm = trimesh.smoothing.filter_laplacian(mesh,lamb=0.01)
    mean_curve = discrete_mean_curvature_measure(mesh, mesh.vertices, 2)/sphere_ball_intersection(1, 2)
    gaussian_curve = discrete_gaussian_curvature_measure(mesh, mesh.vertices, 2)/sphere_ball_intersection(1, 2)
    df_dense['Mean curvature'] = mean_curve
    df_dense['Gaussian curvature'] = gaussian_curve
    
    #----- Use manual 3D topology to compute neighborhoods -----
    # Load the actual neighborhood topology
    A = np.load(path.join(dirname,f'Image flattening/flat_adj/adjmat_t{t}.npy'))
    D = distance.squareform(distance.pdist(dense_coords_3d_um))
    
    #----- Use macrophage annotations to find distance to them -----
    #NB: the macrophage coords are in um
    if not DEMO:
        macrophage_xyz = pd.read_csv(path.join(dirname,f'3d_cyto_seg/macrophages/t{t}.csv'))
        macrophage_xyz = macrophage_xyz.rename(columns={'axis-0':'Z','axis-1':'Y','axis-2':'X'})
        macrophage_xyz['X'] = macrophage_xyz['X'] * dx
        macrophage_xyz['Y'] = macrophage_xyz['Y'] * dx
        df_dense['Distance to closest macrophage'] = \
            find_distance_to_closest_point(pd.DataFrame(dense_coords_3d_um,columns=['Z','Y','X']), macrophage_xyz)
    
    # Propagate differentiation annotations
    A_diff = A.copy()
    A_diff[:,~df_dense['Differentiating']] = 0
    A_diff = A_diff + A_diff.T
    A_diff[A_diff > 1] = 1
    A_planar = A - A_diff
    
    # Resave adjmat as planar v. diff
    if SAVE and not DEMO:
        im = draw_adjmat_on_image(A_planar,dense_coords,[XX,XX])
        io.imsave(path.join(dirname,f'Image flattening/flat_adj/t{t}_planar.tif'),im.astype(np.uint16),check_contrast=False)
        im = draw_adjmat_on_image(A_diff,dense_coords,[XX,XX])
        io.imsave(path.join(dirname,f'Image flattening/flat_adj/t{t}_diff.tif'),im.astype(np.uint16),check_contrast=False)
    
    # Initialize neighborhood stats
    df_dense['Nuclear surface area'] = np.nan
    df_dense['Nuclear axial component'] = np.nan
    df_dense['Nuclear axial angle'] = np.nan
    df_dense['Nuclear planar component 1'] = np.nan
    df_dense['Nuclear planar component 2'] = np.nan
    df_dense['Nuclear axial eccentricity'] = np.nan
    df_dense['Nuclear planar eccentricity'] = np.nan
    df_dense['Nuclear bbox top'] = np.nan
    df_dense['Nuclear bbox bottom'] = np.nan
    
    df_dense['Mean neighbor dist'] = np.nan
    
    df_dense['Mean neighbor nuclear volume'] = np.nan
    df_dense['Std neighbor nuclear volume'] = np.nan
    
    df_dense['Mean neighbor cell volume'] = np.nan
    df_dense['Std neighbor cell volume'] = np.nan
    df_dense['Max neighbor cell volume'] = np.nan
    df_dense['Min neighbor cell volume'] = np.nan
    
    df_dense['Mean neighbor apical area'] = np.nan
    df_dense['Std neighbor apical area'] = np.nan
    df_dense['Mean neighbor basal area'] = np.nan
    df_dense['Std neighbor basal area'] = np.nan
    
    df_dense['Mean neighbor cell height'] = np.nan
    df_dense['Std neighbor cell height'] = np.nan
    
    df_dense['Coronal area'] = np.nan
    df_dense['Coronal angle'] = np.nan
    df_dense['Coronal eccentricity'] = np.nan
    
    df_dense['Mean neighbor height from BM'] = np.nan
    df_dense['Max neighbor height from BM'] = np.nan
    df_dense['Min neighbor height from BM'] = np.nan
    
    df_dense['Mean neighbor collagen alignment'] = np.nan
    
    # df_dense['Mean planar neighbor height from BM'] = np.nan
    # df_dense['Mean diff neighbor height from BM'] = np.nan
    
    # Load basal masks for current frame
    frame_basal_mask = io.imread(path.join(dirname,f'Image flattening/basal_masks/t{t}.tif'))
    # Make local neighborhood measurement stats
    props = measure.regionprops(nuc_dense_seg,extra_properties = [surface_area])
    for i,this_cell in df_dense.iterrows(): #NB: i needs to be 0-index
        
        bbox = props[i]['bbox']
        df_dense['Nuclear bbox top'] = bbox[0]
        df_dense['Nuclear bbox bottom'] = bbox[3]
    
        I = props[i]['inertia_tensor']
        SA = props[i]['surface_area'] * dx**2
        Iaxial,phi,Ia,Ib,theta = parse_3D_inertial_tensor(I)
        df_dense.at[i,'Nuclear surface area'] = SA  * dx**2 # currentlly the function doesn't native use spacing
        df_dense.at[i,'Nuclear axial component'] = Iaxial
        df_dense.at[i,'Nuclear axial angle'] = phi
        df_dense.at[i,'Nuclear planar component 1'] = Ia
        df_dense.at[i,'Nuclear planar component 2'] = Ib
        df_dense.at[i,'Nuclear axial eccentricity'] = Ia/Iaxial
        df_dense.at[i,'Nuclear planar eccentricity'] = Ib/Ia
        df_dense.at[i,'Nuclear planar orientation'] = theta
        
        # Use neighbor matrices
        planar_neighbor_idx = np.where(A_planar[i,:])[0]
        diff_neighbor_idx = np.where(A_diff[i,:])[0]
        df_dense.at[i,'Num planar neighbors'] = len(planar_neighbor_idx)
        df_dense.at[i,'Num diff neighbors'] = len(diff_neighbor_idx)
        
        # All neighbors
        all_neighbor_idx = np.where(A[i,:])[0]
        if len(all_neighbor_idx) > 0:
            
            # Estimate from neighbor identity alone
            fucci_intensities = df_dense.loc[np.hstack([diff_neighbor_idx,planar_neighbor_idx])]['FUCCI bg sub']
            df_dense.at[i,'Mean neighbor FUCCI intensity'] = fucci_intensities.mean()
            fucci_category = df_dense.loc[np.hstack([diff_neighbor_idx,planar_neighbor_idx])]['FUCCI thresholded'] == 'High'
            df_dense.at[i,'Frac neighbor FUCCI high'] = fucci_category.sum()/len(fucci_category)
            
            neighbor_heights = df_dense.loc[all_neighbor_idx]['Height to BM']
            df_dense.at[i,'Mean neighbor height from BM'] = neighbor_heights.mean()
            df_dense.at[i,'Std neighbor height from BM'] = neighbor_heights.std()
            df_dense.at[i,'Max neighbor height from BM'] = neighbor_heights.max()
            df_dense.at[i,'Min neighbor height from BM'] = neighbor_heights.min()
            
            # Distance to neighbors
            neighbor_dists = D[i, all_neighbor_idx]
            df_dense.at[i,'Mean neighbor dist'] = neighbor_dists.mean()
            
            # Neighbor size/shape info -- nuclear
            df_dense.at[i,'Mean neighbor nuclear volume'] = df_dense.iloc[all_neighbor_idx]['Nuclear volume'].mean()
            df_dense.at[i,'Std neighbor nuclear volume'] = df_dense.iloc[all_neighbor_idx]['Nuclear volume'].std()
            df_dense.at[i,'Max neighbor nuclear volume'] = df_dense.iloc[all_neighbor_idx]['Nuclear volume'].max()
            df_dense.at[i,'Min neighbor nuclear volume'] = df_dense.iloc[all_neighbor_idx]['Nuclear volume'].min()
            # df_dense.at[i,'Mean neighbor nuclear volume normalized'] = df_dense.iloc[all_neighbor_idx]['Nuclear volume normalized'].mean()
            # df_dense.at[i,'Std neighbor nuclear volume normalized'] = df_dense.iloc[all_neighbor_idx]['Nuclear volume normalized'].std()
            # df_dense.at[i,'Max neighbor nuclear volume normalized'] = df_dense.iloc[all_neighbor_idx]['Nuclear volume normalized'].max()
            # df_dense.at[i,'Min neighbor nuclear volume normalized'] = df_dense.iloc[all_neighbor_idx]['Nuclear volume normalized'].min()
            
            # Neighbor size/shape info -- cortical
            df_dense.at[i,'Mean neighbor cell volume'] = df_dense.iloc[all_neighbor_idx]['Cell volume'].mean()
            df_dense.at[i,'Std neighbor cell volume'] = df_dense.iloc[all_neighbor_idx]['Cell volume'].std()
            df_dense.at[i,'Max neighbor cell volume'] = df_dense.iloc[all_neighbor_idx]['Cell volume'].max()
            df_dense.at[i,'Min neighbor cell volume'] = df_dense.iloc[all_neighbor_idx]['Cell volume'].min()
            df_dense.at[i,'Mean neighbor apical area'] = df_dense.iloc[all_neighbor_idx]['Apical area'].mean()
            df_dense.at[i,'Std neighbor apical area'] = df_dense.iloc[all_neighbor_idx]['Apical area'].std()
            df_dense.at[i,'Mean neighbor basal area'] = df_dense.iloc[all_neighbor_idx]['Basal area'].mean()
            df_dense.at[i,'Std neighbor basal area'] = df_dense.iloc[all_neighbor_idx]['Basal area'].std()
            
            df_dense.at[i,'Mean neighbor cell height'] = df_dense.iloc[all_neighbor_idx]['Height'].max()
            df_dense.at[i,'Std neighbor cell height'] = df_dense.iloc[all_neighbor_idx]['Height'].min()
            
            df_dense.at[i,'Mean neighbor collagen alignment'] = df_dense.iloc[all_neighbor_idx]['Collagen alignment'].mean()
    
            # If this is a 'dividing cell of interest', then estimate 'corona' using full neighbor cyto seg
            if not np.isnan(this_cell['basalID']):
                
                # Get the cytoIDs
                all_neighbor_cytoIDs = df_dense.iloc[all_neighbor_idx]['CytoID']
                all_neighbor_cytoIDs = nonans(all_neighbor_cytoIDs)
                if len(all_neighbor_cytoIDs) > 0:
                    basal_mask = np.zeros_like(frame_basal_mask,dtype=bool)
                    for ID in all_neighbor_cytoIDs:
                        basal_mask = basal_mask | (frame_basal_mask == ID)
                    p = measure.regionprops(basal_mask.astype(int))[0]
                    df_dense.at[i,'Coronal area'] = p['area'] * dx**2
                    df_dense.at[i,'Coronal eccentricity'] = p['eccentricity']
                    theta = np.rad2deg(p['orientation'])
                    if theta < 0: # Constrain angle to be in quad I and IV
                        theta = theta + 180
                    df_dense.at[i,'Coronal angle'] = theta
                    
            else:
                #-- Below block is estimating from nuclear posiition, which is probably worse
                X = dense_coords[planar_neighbor_idx,1]
                Y = dense_coords[planar_neighbor_idx,0]
                if len(X) > 2:
                    order = argsort_counter_clockwise(X,Y)
                    X = X[order]
                    Y = Y[order]
                    im = np.zeros([XX,XX])
                    rr,cc = draw.polygon(Y,X)
                    im[rr,cc] = 1
                    p = measure.regionprops(im.astype(int))[0]
                    df_dense.at[i,'Coronal area'] = p['area']  * dx**2
                    df_dense.at[i,'Coronal eccentricity'] = p['eccentricity']
                    theta = np.rad2deg(p['orientation'])
                    if theta < 0:
                        theta = theta + 180
                    df_dense.at[i,'Coronal angle'] = theta
                
    # Save the DF
    df.append(df_dense)
    
    #Save a bunch of intermediates
    # Save segmentation with text labels @ centroid
    # im_cellposeID = draw_labels_on_image(dense_coords,df_dense['basalID'],[XX,XX],font_size=12)
    # im_cellposeID.save(path.join(dirname,f'3d_nuc_seg/cellposeIDs/t{t}.tif'))
    
    df_dense_ = df_dense.loc[ ~np.isnan(df_dense['basalID']) ]
    if SAVE and not DEMO:
        colorized = colorize_segmentation(nuc_dense_seg,
                                          {k:v for k,v in zip(df_dense['CellposeID'].values,df_dense['Differentiating'].values)})
        io.imsave(path.join(dirname,f'3d_nuc_seg/Differentiating/t{t}.tif'),colorized.astype(np.int8),check_contrast=False)
    
df = pd.concat(df,ignore_index=True)

if SAVE:
    
    df.to_csv(path.join(dirname,'tissue_dataframe.csv'))
    print(f'Saved to: {dirname}')

