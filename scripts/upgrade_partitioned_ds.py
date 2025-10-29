# Needs to be executed from within the scripts directory
# wget -r -np -nH --cut-dirs=1 -R "index.html*" -q https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/sdss/pdr3_dud_22.5/healpix=1175/
# wget -r -np -nH --cut-dirs=1 -R "index.html*" -q https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/sdss/sdss.py
# wget -r -np -nH --cut-dirs=1 -R "index.html*" -q https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/sdss/sdss/healpix=1175/
# wget -r -np -nH --cut-dirs=1 -R "index.html*" -q https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/sdss/sdss.py
from datasets import load_dataset_builder, Dataset
from mmu.utils import get_catalog
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from huggingface_hub import create_repo

# Load the dataset descriptions from local copy of the data
print("Loading SDSS builder...")
sdss_builder = load_dataset_builder("data/MultimodalUniverse/v1/sdss", trust_remote_code=True)

print("Downloading and preparing dataset...")
sdss_builder.download_and_prepare()

print("Extracting catalog for _index partition...")
# Get catalog with only the essential columns for the index

sdss_catalog = get_catalog(sdss_builder, keys=['object_id', 'ra', 'dec', 'healpix'])

print("Loading full dataset...")

def match_sdss_catalog_object_ids(example, catalog):
    example_obj_id = example['object_id'].strip("b'")
    catalog_entry = catalog[catalog['object_id'] == example_obj_id]
    assert len(catalog_entry) == 1
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
output_dir = Path("output_dataset")
output_dir.mkdir(exist_ok=True)

# Save the _index partition (not partitioned by healpix)
print("Saving _index partition...")
index_path = output_dir / "_index"
index_path.mkdir(exist_ok=True)
index_dataset.to_parquet(str(index_path / "index.parquet"))

# Save the main dataset partitioned by healpix
print("Saving main dataset partitioned by healpix...")
# Group by healpix and save each partition
for split, healpix_list in sdss_dataset.unique('healpix').items():
    for hp_value in healpix_list:
        print(f"  Processing healpix={hp_value}...")
        # Filter dataset for this healpix value
        hp_dataset = sdss_dataset[split].filter(lambda x: x['healpix'] == hp_value)

        # Create healpix partition directory
        hp_dir = output_dir / f"healpix={hp_value}"
        hp_dir.mkdir(exist_ok=True)

        # Save partition - convert to arrow table first
        memory_mapped_table = hp_dataset.data
        target_path = Path( hp_dir / split )
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

repo_id = "TobiasPitters/mmu-sdss-partitioned"

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
    folder_path=str(output_dir),
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
