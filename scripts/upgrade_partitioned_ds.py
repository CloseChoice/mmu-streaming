# Execute with datasets==3.6 && numpy<2
# Needs to be executed from within the scripts directory
# wget -r -np -nH --cut-dirs=1 -R "index.html*" -q https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/sdss/pdr3_dud_22.5/healpix=1175/
# wget -r -np -nH --cut-dirs=1 -R "index.html*" -q https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/sdss/sdss.py
# wget -r -np -nH --cut-dirs=1 -R "index.html*" -q https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/sdss/sdss/healpix=1175/
# wget -r -np -nH --cut-dirs=1 -R "index.html*" -q https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/sdss/sdss.py

import subprocess
from datasets import load_dataset_builder, Dataset
from mmu.utils import get_catalog
import pyarrow.parquet as pq
from pathlib import Path
from huggingface_hub import create_repo
import shutil
import numpy as np

# ============================================================================
# Configuration: Specify healpixels to download
# ============================================================================
# Dataset URL structure mapping
DATASET_URL_PATHS = {
    "sdss": "sdss",
    "hsc": "pdr3_dud_22.5",
}

# HEALPIXELS_TO_DOWNLOAD = [1172, 1173, 1174, 1175]  # Add the healpixels you want here
# DATASET = "sdss"
DATASET = "hsc"
HEALPIXELS_TO_DOWNLOAD = [1172, 1173]  # Add the healpixels you want here
BASE_URL = f"https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/{DATASET}"
LOCAL_DATA_DIR = Path(f"data/MultimodalUniverse/v1/{DATASET}")
DATASET_SUBDIR = DATASET_URL_PATHS[DATASET]

# ============================================================================
# Download missing healpixel data
# ============================================================================
print("Checking for missing healpixel data...")
for hp in HEALPIXELS_TO_DOWNLOAD:
    hp_dir = LOCAL_DATA_DIR / DATASET_SUBDIR / f"healpix={hp}"

    if hp_dir.exists() and any(hp_dir.iterdir()):
        print(f"  healpix={hp}: already exists, skipping download")
    else:
        # import pdb; pdb.set_trace()
        print(f"  healpix={hp}: downloading...")
        # Download the healpix partition
        cmd = [
            "wget", "-r", "-np", "-nH", "--cut-dirs=1",
            "-R", "index.html*", "-q",
            f"{BASE_URL}/{DATASET_SUBDIR}/healpix={hp}/"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"    ✓ Downloaded healpix={hp}")
        else:
            print(f"    ✗ Failed to download healpix={hp}: {result.stderr}")

# Download sdss.py if not present
sdss_py = LOCAL_DATA_DIR / f"{DATASET}.py"
if not sdss_py.exists():
    print("Downloading sdss.py...")
    cmd = [
        "wget", "-r", "-np", "-nH", "--cut-dirs=1",
        "-R", "index.html*", "-q",
        f"{BASE_URL}/{DATASET}.py"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("  ✓ Downloaded sdss.py")
    else:
        print(f"  ✗ Failed to download sdss.py: {result.stderr}")

print("Download check complete!\n")

# ============================================================================
# Clean up healpixel partitions not in the list
# ============================================================================
print("Cleaning up unwanted healpixel partitions...")
sdss_data_dir = LOCAL_DATA_DIR / DATASET_SUBDIR
if sdss_data_dir.exists():
    for item in sdss_data_dir.iterdir():
        if item.is_dir() and item.name.startswith("healpix="):
            hp_value = int(item.name.split("=")[1])
            if hp_value not in HEALPIXELS_TO_DOWNLOAD:
                print(f"  Removing healpix={hp_value} (not in specified list)...")
                shutil.rmtree(item)
                print(f"    ✓ Removed healpix={hp_value}")
print("Cleanup complete!\n")

# Load the dataset descriptions from local copy of the data
print("Loading SDSS builder...")
sdss_builder = load_dataset_builder(f"data/MultimodalUniverse/v1/{DATASET}", trust_remote_code=True)

# Clear cache to ensure new healpixels are picked up
print("Clearing dataset cache...")
import shutil
cache_dir = Path(sdss_builder.cache_dir)
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print(f"  ✓ Cleared cache at {cache_dir}")

print("Downloading and preparing dataset...")
sdss_builder.download_and_prepare()

print("Extracting catalog for _index partition...")
# Get catalog with only the essential columns for the index

sdss_catalog = get_catalog(sdss_builder, keys=['object_id', 'ra', 'dec', 'healpix'])

print("Loading full dataset...")

def match_sdss_catalog_object_ids(example, catalog):
    example_obj_id = example['object_id'].strip("b'")
    if isinstance(example_obj_id, str) and isinstance(catalog['object_id'][0], str):
        example_obj_id = example_obj_id
    elif np.issubdtype(catalog['object_id'][0], np.integer):
        example_obj_id = int(example_obj_id)
    catalog_entry = catalog[catalog['object_id'] == example_obj_id]
    try:
        assert len(catalog_entry) == 1
    except Exception:
        import pdb; pdb.set_trace()
    # return {**example, 'ra': catalog_entry['ra'][0], 'dec': catalog_entry['dec'][0], 'healpix': catalog_entry['healpix'][0]}
    return {**example, 'healpix': catalog_entry['healpix'][0]}

sdss_dataset = sdss_builder.as_dataset().map(lambda example: match_sdss_catalog_object_ids(example, sdss_catalog))

# Convert catalog to Dataset for the _index partition
print("Creating _index dataset...")
catalog_dict = {
    'object_id': sdss_catalog['object_id'],
    'ra': sdss_catalog['ra'],
    'dec': sdss_catalog['dec'],
    'healpix': sdss_catalog['healpix']
}
index_dataset = Dataset.from_dict(catalog_dict)

# Create output directory structure
upload_path = Path(f"output_dataset/{DATASET}")
output_dir = upload_path / "train"
output_dir.mkdir(parents=True, exist_ok=True)

# Save the _index partition (not partitioned by healpix)
print("Saving _index partition...")
index_path = output_dir / "_index"
index_path.mkdir(exist_ok=True)
index_dataset.to_parquet(str(index_path / "index.parquet"))

# Save the main dataset partitioned by healpix
print("Saving main dataset partitioned by healpix...")
# Group by healpix and save each partition
for split, healpix_list in sdss_dataset.unique('healpix').items():
    # todo: we ignore the split in here! So we implicitly imply there is only one and this is train!
    for hp_value in healpix_list:
        print(f"  Processing healpix={hp_value}...")
        # Filter dataset for this healpix value
        hp_dataset = sdss_dataset[split].filter(lambda x: x['healpix'] == hp_value)

        # Create healpix partition directory
        hp_dir = output_dir / f"healpix={hp_value}"
        hp_dir.mkdir(exist_ok=True)

        # Save partition - convert to arrow table first
        memory_mapped_table = hp_dataset.data
        target_path = Path( hp_dir )
        target_path.mkdir(exist_ok=True)
        pq.write_table(memory_mapped_table.table, target_path / "data.parquet")
    # pq.write_table(arrow_table, hp_dir / "data.parquet")

print("Dataset preparation complete!")
print(f"Output saved to: {output_dir}")

# Now upload to HuggingFace Hub
# We'll upload the index and data partitions separately then combine them
print("\n" + "="*60)
print("Uploading to HuggingFace Hub...")
print("="*60)

repo_id = f"TobiasPitters/mmu-{DATASET}-partitioned"

# Option 1: Upload the entire directory structure
# This preserves the _index and healpix= partition structure
from huggingface_hub import HfApi
api = HfApi()

print(f"\nUploading complete dataset structure to {repo_id}...")
create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    exist_ok=True,
)
api.upload_folder(
    folder_path=str(upload_path),
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo=".",
)

print(f"\n✓ Upload complete!")
print(f"Dataset available at: https://huggingface.co/datasets/{repo_id}")
print("\nTo load the dataset:")
print(f"  from datasets import load_dataset")
print(f"  # Load full data (all healpix partitions)")
print(f"  ds = load_dataset('{repo_id}', split='train')")
print(f"  # Load only index")
print(f"  index = load_dataset('{repo_id}', data_dir='_index', split='train')")
print(f"  # Load specific healpix partition")
print(f"  ds_hp = load_dataset('{repo_id}', data_dir='healpix=1175', split='train')")
