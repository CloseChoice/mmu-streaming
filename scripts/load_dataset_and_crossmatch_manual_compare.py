# Prerequisites:
# uv pip install -r requirements.txt
# ./download_sdss_hsc.sh
from datasets import load_dataset
from functions.crossmatch_manual import cross_match_datasets_manual


sdss = load_dataset("TobiasPitters/mmu-sdss-with-coordinates")
hsc = load_dataset("TobiasPitters/mmu-hsc-with-coordinates")

matched = cross_match_datasets_manual(sdss,
                                      hsc,
                                      left_name="sdss",
                                      right_name="hsc",
                                      matching_radius=1.0,
                                      # well, coordinate_columns might not be the best name here
                                      coordinate_columns=['ra', 'dec', 'healpix', 'object_id']
                                      )

from datasets import load_dataset_builder, concatenate_datasets
from mmu.utils import cross_match_datasets

# Load the dataset descriptions from local copy of the data
sdss = load_dataset_builder("data/MultimodalUniverse/v1/sdss", trust_remote_code=True)
hsc = load_dataset_builder("data/MultimodalUniverse/v1/hsc", trust_remote_code=True)


# Use the cross matching utility to return a new HF dataset, the intersection
# of the parent samples.
dset = cross_match_datasets(sdss, # Left dataset
                            hsc,  # Right dataset
                            matching_radius=1.0, # Distance in arcsec
                            )
for example in dset:
    obj_id = example['object_id']
    matched_obj = matched.filter(lambda x: x['object_id'] == obj_id)[0]
    # drop additional columns we created for cross-matching
    matched_obj.pop('ra')
    matched_obj.pop('dec')
    matched_obj.pop('healpix')
    assert example == matched_obj
