import numpy as np
import pandas as pd
from skimage import io
from os import path

#--- Bookkeepers ---
from imageUtils import trim_multimasks_to_shared_bounding_box
from aicsshparam import shtools
import pyvista as pv

def reconstruct_mesh(coeffs,lmax=5):
    '''
    Input: coeffs in dict format as output by aicspharam
    '''
    coeffs = coeffs.to_dict()
    coeffs = {'_'.join(k.split('_')[1:3]):v for k,v in coeffs.items()}
    # Convert to matrix
    mat = np.zeros((2, lmax + 1, lmax + 1), dtype=np.float32)
    for L in range(lmax):
        for M in range(L + 1):
            for cid, C in enumerate(["C", "S"]):
                key = f"shcoeffs_L{L}M{M}{C}"
                if key in coeffs.keys():
                    mat[cid, L, M] = coeffs[key]
                else:
                    mat[cid,L,M] = 0
    mesh = shtools.get_even_reconstruction_from_coeffs(mat)
    return mesh


def plot_cells_side_by_side(cells,num_cols=None):
    '''
    Plots a list of cells (cell['cyto'] and cell['nuc'] are mesh objects) using pyvista
    num_grid controls the number of cells on one row
    '''
    num_cells = len(cells)
    if num_cols is None:
        num_cols = num_cells
    num_rows = num_cells // num_cols

    pl = pv.Plotter(shape=(num_rows,num_cols))
    for i in range(num_cells):
        if i >= num_cells:
            continue
        pl.subplot( i // num_cols ,i % num_cols)
        if 'nuc' in cells[i]:
            pl.add_mesh(pv.wrap(cells[i]['nuc']),color='r', opacity=0.5)
        pl.add_mesh(pv.wrap(cells[i]['cyto']),color='y', opacity=0.4)
        pl.add_axes()
    pl.link_views()
    pl.view_isometric()
    pv.Line((800, 400, 0), (800, 400, 20))
    pl.show()

def reconstruct_mesh_from_averaged_coeffs(index,coeffs,cyto_col,nuc_col):
    '''

    '''
    mean_coeffs = coeffs.loc[index].mean()
    mean_cyto = mean_coeffs[cyto_col]
    mean_nuc = mean_coeffs[nuc_col]
    print(type(mean_coeffs))
    avg_cell = {'cyto':reconstruct_mesh(mean_cyto)[0],
                'nuc':reconstruct_mesh(mean_nuc)[0],
               }
    return avg_cell

def get_sorted_cell_indexes(df,feature):
    return df.droplevel(axis=1,level=1).dropna(subset=feature).sort_values(feature).index


def extract_nuc_and_cell_mask_from_idx(idx : tuple,
                                        tracked_nuc_by_region:dict,
                                        tracked_cyto_by_region:dict,):
    '''
    Returns a tuple of nuc_mask,cyto_mask if given the measurement index of the cell.
    Index should be in the format (frame,'Region_trackID'), where frame is int

    '''
    assert(len(idx)) == 2

    frame = idx[0]
    region,trackID = idx[1].split('_')
    trackID = int(trackID)
    nuc_mask = tracked_nuc_by_region[region][frame,...] == trackID
    cyto_mask = tracked_cyto_by_region[region][frame,...] == trackID
    nuc_mask,cyto_mask = trim_multimasks_to_shared_bounding_box((nuc_mask,cyto_mask))

    return nuc_mask,cyto_mask

def get_microenvironment_mask(trackID,
                              adjdict: dict,
                              cyto_seg: np.array):
    adjacentIDs = adjdict[trackID]
    mask = np.zeros_like(cyto_seg,dtype=bool)
    for ID in adjacentIDs:
        mask[cyto_seg == ID] = True

    return mask

def extract_nuc_and_cell_and_microenvironment_mask_from_idx(idx : tuple,
                                        adjdict_by_region:dict,
                                        tracked_nuc_by_region:dict,
                                        tracked_cyto_by_region:dict,
                                        trim=True,
                                        cell_type='Basal'):
    '''
    Returns a tuple of nuc_mask,cyto_mask,microenvironment_mask
    if given the measurement index of the cell.

    Index should be in the format (frame,'Region_trackID'), where frame is int

    '''

    assert(len(idx)) == 2

    frame = idx[0]
    region,trackID = idx[1].split('_')
    trackID = int(trackID)
    nuc_mask = tracked_nuc_by_region[region][frame,...] == trackID
    if cell_type == 'Basal':
        cyto_mask = tracked_cyto_by_region[region][frame,...] == trackID
        microenvironment_mask = get_microenvironment_mask(trackID,adjdict_by_region[region][frame],
                                                          tracked_cyto_by_region[region][frame,...])
    else:
        cyto_mask = np.zeros_like(nuc_mask)
        microenvironment_mask = np.zeros_like(nuc_mask)
    if trim:
        nuc_mask,cyto_mask,microenvironment_mask = trim_multimasks_to_shared_bounding_box((nuc_mask,cyto_mask,microenvironment_mask))

    return nuc_mask,cyto_mask,microenvironment_mask

def get_nuc_and_cell_and_microenvironment_movie(trackID,
                                        df:pd.DataFrame,
                                        adjdict_by_region:dict,
                                        tracked_nuc_by_region:dict,
                                        tracked_cyto_by_region:dict,
                                        standard_size:tuple):

    cell = df.swaplevel(axis=0).loc[trackID,:]
    cell['TrackID'] = trackID
    cell = cell.reset_index().set_index(['Frame','TrackID'])
    indexes = cell.index

    nuc_masks = []
    cell_masks = []
    micro_masks = []

    for idx in indexes:
        # print(idx)
        n,c,m = extract_nuc_and_cell_and_microenvironment_mask_from_idx(idx,adjdict_by_region,
                                                tracked_nuc_by_region,tracked_cyto_by_region,
                                                cell_type = cell.loc[idx]['Cell type','Meta'],
                                                trim=False)
        nuc_masks.append(n)
        cell_masks.append(c)
        micro_masks.append(m)

    trimmed_masks = trim_multimasks_to_shared_bounding_box(nuc_masks+cell_masks+micro_masks)
    trimmed_nuc = trimmed_masks[:len(indexes)]
    trimmed_cell = trimmed_masks[len(indexes)+1:len(indexes)*2]
    trimmed_micro = trimmed_masks[2*len(indexes):]

    return (trimmed_nuc,trimmed_cell,trimmed_micro)
