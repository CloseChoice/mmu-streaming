# NOTE: use datasets==3.6 for this
# Run this first
# uv pip install -r requirements.txt
# ./download_sdss_hsc.sh
from datasets import load_dataset_builder, concatenate_datasets
from mmu.utils import get_catalog

# Load the dataset descriptions from local copy of the data
sdss = load_dataset_builder("data/MultimodalUniverse/v1/sdss", trust_remote_code=True)
hsc = load_dataset_builder("data/MultimodalUniverse/v1/hsc", trust_remote_code=True)

sdss_catalog = get_catalog(sdss)
hsc_catalog = get_catalog(hsc)

def match_sdss_catalog_object_ids(example, catalog):
    example_obj_id = example['object_id'].strip("b'")
    catalog_entry = catalog[catalog['object_id'] == example_obj_id]
    assert len(catalog_entry) == 1
    return {**example, 'ra': catalog_entry['ra'][0], 'dec': catalog_entry['dec'][0], 'healpix': catalog_entry['healpix'][0]}
    
def match_hsc_catalog_object_ids(example, catalog):
    example_obj_id = int(example['object_id'])
    catalog_entry = catalog[catalog['object_id'] == example_obj_id]
    assert len(catalog_entry) == 1
    return {**example, 'ra': catalog_entry['ra'][0], 'dec': catalog_entry['dec'][0], 'healpix': catalog_entry['healpix'][0]}

sdss_train = sdss.as_dataset(split="train")
sdss_mapped = sdss_train.map(lambda example: match_sdss_catalog_object_ids(example, sdss_catalog))
sdss_mapped.push_to_hub("TobiasPitters/mmu-sdss-with-coordinates", split="train")

# hsc_mapped = hsc.as_dataset().map(lambda example: match_hsc_catalog_object_ids(example, hsc_catalog))
# hsc_mapped.push_to_hub("TobiasPitters/mmu-hsc-with-coordinates")
