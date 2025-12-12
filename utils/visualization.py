import numpy as np
import pandas as pd
from skimage import io
from os import path

#--- Bookkeepers ---
from imageUtils import trim_multimasks_to_shared_bounding_box

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
