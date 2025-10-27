from datasets import Dataset
from typing import List
from astropy.table import Table, hstack, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np


def cross_match_datasets_manual(
                         left_ds : Dataset, 
                         right_ds : Dataset,
                         left_name: str,
                         right_name: str,
                         cache_dir : str = None,
                         keep_in_memory : bool = False,
                         matching_radius : float = 1., 
                         return_catalog_only : bool = False,
                         num_proc : int = None,
                         coordinate_columns : List[str] = None
):
    left = Table.from_pandas(left_ds['train'].to_pandas())
    right = Table.from_pandas(right_ds['train'].to_pandas())
    if coordinate_columns is not None:
        left = left[coordinate_columns]
        right = right[coordinate_columns]

    left['sc'] = SkyCoord(left['ra'], 
                          left['dec'], unit='deg')
    
    right['sc'] = SkyCoord(right['ra'],
                           right['dec'], unit='deg')

    cat_left = left
    cat_right = right
    # Cross match the catalogs and restricting them to matches
    idx, sep2d, _ = cat_left['sc'].match_to_catalog_sky(cat_right['sc'])
    mask = sep2d < matching_radius*u.arcsec
    cat_left = cat_left[mask]
    cat_right = cat_right[idx[mask]]
    assert len(cat_left) == len(cat_right), "There was an error in the cross-matching."
    print("Initial number of matches: ", len(cat_left))
    matched_catalog = hstack([cat_left, cat_right], 
                             table_names=[left_name, right_name],
                             uniq_col_name='{table_name}_{col_name}')
    # Remove objects that were matched between the two catalogs but fall under different healpix indices
    mask = matched_catalog[f'{left_name}_healpix'] == matched_catalog[f'{right_name}_healpix']
    matched_catalog = matched_catalog[mask]
    print("Number of matches lost at healpix region borders: ", len(cat_left) - len(matched_catalog))
    print("Final size of cross-matched catalog: ", len(matched_catalog))

    # Adding default columns to respect format
    matched_catalog['object_id'] = matched_catalog[left_name+'_object_id']
    matched_catalog['ra'] = 0.5*(matched_catalog[left_name+'_ra'] +
                                 matched_catalog[right_name+'_ra'])
    matched_catalog['dec'] = 0.5*(matched_catalog[left_name+'_dec'] +
                                 matched_catalog[right_name+'_dec'])
    
    # Check that all matches have the same healpix index
    assert np.all(matched_catalog[left_name+'_healpix'] == matched_catalog[right_name+'_healpix']), "There was an error in the cross-matching."
    matched_catalog['healpix'] = matched_catalog[left_name+'_healpix']
    matched_catalog = matched_catalog.group_by(['healpix'])

    if return_catalog_only:
        return matched_catalog

    catalog_groups = [group for group in matched_catalog.groups]
    # Create a generator function that merges the two generators
    def _generate_examples(groups):
        for group in groups:
            # can be used to filter the correct files, maybe we'll need a datasetbuilder either wway
            healpix = group['healpix'][0]
            # this is probably not easily streamable
            left_ds_filtered = left_ds['train'].filter(lambda example: example['object_id'] in group[f'{left_name}_object_id'])
            right_ds_filtered = right_ds['train'].filter(lambda example: example['object_id'] in group[f'{right_name}_object_id'])
            for i, example_left in enumerate(left_ds_filtered):
                right_obj_id = group[group[f'{left_name}_object_id'] == example_left['object_id']][f'{right_name}_object_id'][0]
                example_right = right_ds_filtered.filter(lambda example: example['object_id'] == right_obj_id)[0]
                assert str(group[i][left_name+'_object_id']) in example_left['object_id'], "There was an error in the cross-matching generation."
                assert str(group[i][right_name+'_object_id']) in example_right['object_id'], "There was an error in the cross-matching generation."
                example_left.update(example_right)
                yield example_left
    
    # Merging the features of both datasets
    features = left_ds['train'].features.copy()
    features.update(right_ds['train'].features)

    # Generating a description for the new dataset based on the two parent datasets
    description = (f"Cross-matched dataset between {left_name} and {right_name}.")
    
    # Create the new dataset
    return Dataset.from_generator(_generate_examples,
                                                   features,
                                                   cache_dir=cache_dir,
                                                   gen_kwargs={'groups':catalog_groups},
                                                   num_proc=num_proc,
                                                   keep_in_memory=keep_in_memory,
                                                   description=description)
