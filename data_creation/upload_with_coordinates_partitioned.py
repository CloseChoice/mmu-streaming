#!/usr/bin/env python3
# NOTE: use datasets==3.6 for this
# Run with: uv run --with datasets==3.6 python data_creation/upload_with_coordinates_partitioned.py

from datasets import load_dataset_builder, Dataset
from mmu.utils import get_catalog
import pyarrow.parquet as pq
from pathlib import Path
from huggingface_hub import create_repo, HfApi

# ============================================================================
# Configuration
# ============================================================================
DATASET = "sdss"  # or "hsc"
LOCAL_DATA_DIR = f"data/MultimodalUniverse/v1/{DATASET}"
REPO_ID = f"TobiasPitters/mmu-{DATASET}-with-coordinates"

# ============================================================================
# Load dataset and catalog
# ============================================================================
builder = load_dataset_builder(LOCAL_DATA_DIR, trust_remote_code=True)
catalog = get_catalog(builder)

# ============================================================================
# Map function to add coordinates
# ============================================================================
def match_sdss_catalog_object_ids(example, catalog):
    example_obj_id = example['object_id'].strip("b'")
    catalog_entry = catalog[catalog['object_id'] == example_obj_id]
    assert len(catalog_entry) == 1
    return {
        **example,
        'ra': catalog_entry['ra'][0],
        'dec': catalog_entry['dec'][0],
        'healpix': catalog_entry['healpix'][0]
    }

def match_hsc_catalog_object_ids(example, catalog):
    example_obj_id = int(example['object_id'])
    catalog_entry = catalog[catalog['object_id'] == example_obj_id]
    assert len(catalog_entry) == 1
    return {
        **example,
        'ra': catalog_entry['ra'][0],
        'dec': catalog_entry['dec'][0],
        'healpix': catalog_entry['healpix'][0]
    }

# Choose the right matching function
if DATASET == "sdss":
    match_fn = match_sdss_catalog_object_ids
elif DATASET == "hsc":
    match_fn = match_hsc_catalog_object_ids
else:
    raise ValueError(f"Unknown dataset: {DATASET}")

# ============================================================================
# Load and map dataset
# ============================================================================
dataset_train = builder.as_dataset(split="train")
dataset_mapped = dataset_train.map(lambda example: match_fn(example, catalog))

# ============================================================================
# Create output directory structure
# ============================================================================
upload_path = Path(f"output_dataset/{DATASET}_with_coords")
output_dir = upload_path / "train"
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Save dataset partitioned by healpix
# ============================================================================
unique_healpix = dataset_mapped.unique('healpix')

for split, healpix_list in unique_healpix.items():
    for hp_value in healpix_list:
        hp_dataset = dataset_mapped.filter(lambda x: x['healpix'] == hp_value)
        hp_dir = output_dir / f"healpix={hp_value}"
        hp_dir.mkdir(exist_ok=True)
        memory_mapped_table = hp_dataset.data
        pq.write_table(memory_mapped_table.table, hp_dir / "data.parquet")

# ============================================================================
# Upload to HuggingFace Hub
# ============================================================================
api = HfApi()

create_repo(
    repo_id=REPO_ID,
    repo_type="dataset",
    exist_ok=True,
)

api.upload_folder(
    folder_path=str(upload_path),
    repo_id=REPO_ID,
    repo_type="dataset",
    path_in_repo=".",
)
