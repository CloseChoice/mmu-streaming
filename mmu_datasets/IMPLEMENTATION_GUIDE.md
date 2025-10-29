# MMU Dataset Builder - Implementation Guide

This guide helps you implement the custom dataset builder for Multimodal Universe datasets with efficient crossmatching.

## Overview

The builder inherits from HuggingFace's `ArrowBasedBuilder` and adds:
1. Index-first loading strategy
2. Partition filtering based on crossmatching
3. Only downloading relevant data partitions

## Dataset Structure

```
TobiasPitters/mmu-sdss-partitioned/
└── train/                                 # Split directory
    ├── _index/                           # Lightweight index
    │   └── index.parquet                # (object_id, ra, dec, healpix, object_group_id)
    ├── healpix=1172/                     # Spatial partition
    │   └── train/
    │       └── data.parquet             # Full dataset objects
    └── healpix=1173/
        └── train/
            └── data.parquet
```

## Core Classes

### 1. MMUConfig

Configuration dataclass that extends `BuilderConfig`:

```python
@dataclass
class MMUConfig(BuilderConfig):
    # Standard fields (inherited)
    name: str
    data_dir: Optional[str]

    # MMU-specific fields
    split_name: str = "train"              # Split directory name
    index_partition: str = "_index"        # Index directory name
    matching_datasets: Optional[Dict[str, str]] = None    # For crossmatching
    matching_fn: Optional[Callable] = None                # Matching function
    matching_config: Optional[Dict[str, Any]] = None      # Matching params
    columns: Optional[List[str]] = None                   # Column selection
```

### 2. MMUDatasetBuilder

Custom builder that extends `ArrowBasedBuilder`:

```python
class MMUDatasetBuilder(ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = MMUConfig

    def _info(self) -> DatasetInfo:
        """Return dataset metadata and schema."""
        pass

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:
        """Main loading logic."""
        pass

    def _generate_tables(self, files, index):
        """Yield tables from parquet files."""
        pass
```

## Implementation Steps

### Step 1: Implement `_info()`

Returns dataset metadata:

```python
def _info(self) -> DatasetInfo:
    return DatasetInfo(
        features=self.config.features,
        description="MMU dataset with crossmatching support",
    )
```

### Step 2: Implement `_split_generators()`

This is the main method that implements the loading logic:

```python
def _split_generators(self, dl_manager: DownloadManager) -> List[SplitGenerator]:
    # 1. Download and load index partition
    index_table = self._download_and_load_index(dl_manager)

    # 2. Determine which partitions to load
    if self.config.matching_fn is not None:
        # Apply crossmatching
        partitions = self._apply_crossmatch(dl_manager, index_table)
    else:
        # Load all partitions
        partitions = self._get_all_partitions(index_table)

    # 3. Build list of data files to download
    data_files = self._build_data_files_list(partitions)

    # 4. Download data files
    downloaded_files = dl_manager.download(data_files)

    # 5. Return split generator
    return [
        SplitGenerator(
            name="train",
            gen_kwargs={
                "files": downloaded_files,
                "index": index_table
            }
        )
    ]
```

### Step 3: Implement Helper Methods

#### `_apply_crossmatch()`

```python
def _apply_crossmatch(self, dl_manager, index_table):
    # Load indices from other datasets
    other_indices = {}
    for name, dataset_path in self.config.matching_datasets.items():
        other_indices[name] = self._load_other_index(dl_manager, dataset_path)

    # Call user's matching function
    partitions_dict = self.config.matching_fn(
        index_table,
        other_indices,
        self.config.matching_config or {}
    )

    # Return partitions for this (primary) dataset
    return partitions_dict['primary']
```

#### `_get_all_partitions()`

```python
def _get_all_partitions(self, index_table):
    # Extract unique (healpix, object_group_id) combinations
    healpix_col = index_table['healpix'].to_pylist()
    group_col = index_table['object_group_id'].to_pylist()

    unique_partitions = set(zip(healpix_col, group_col))
    return list(unique_partitions)
```

#### `_build_data_files_list()`

```python
def _build_data_files_list(self, partitions):
    # List all files in repo
    all_files = self._list_repository_files()

    # Filter for matching partitions
    data_files = []
    for healpix, group in partitions:
        pattern = f"{self.config.split_name}/healpix={healpix}/train/data.parquet"
        matching_files = [f for f in all_files if pattern in f]

        # Convert to full URLs
        repo_id = self._extract_repo_id_from_url(self.config.data_dir)
        data_files.extend([
            f"hf://datasets/{repo_id}/{f}" for f in matching_files
        ])

    return data_files
```

### Step 4: Implement `_generate_tables()`

```python
def _generate_tables(self, files, index):
    for i, file_path in enumerate(files):
        # Read parquet file
        table = pq.read_table(file_path, columns=self.config.columns)

        # Optionally cast to expected schema
        if self.info.features is not None:
            table = table_cast(table, self.info.features.arrow_schema)

        yield f"{i}", table
```

## Matching Function

The matching function should have this signature:

```python
def spatial_crossmatch_fn(
    primary_index: pa.Table,
    other_indices: Dict[str, pa.Table],
    config: Dict[str, Any]
) -> Dict[str, List[Tuple[int, int]]]:
    """Crossmatch astronomical catalogs by spatial position.

    Args:
        primary_index: Index table for primary dataset
            Columns: object_id, ra, dec, healpix, object_group_id
        other_indices: Dict of {survey_name: index_table}
        config: Configuration dict (e.g., {'tolerance': 1.0})

    Returns:
        Dict mapping dataset name to list of (healpix, group) tuples:
        {
            'primary': [(1172, 0), (1173, 0)],  # Partitions to load for primary
            'hsc': [(2234, 0), (2235, 0)]       # Partitions to load for HSC
        }
    """
    # 1. Extract coordinates from indices
    # 2. Perform spatial crossmatching (e.g., with astropy)
    # 3. For matched objects, determine their (healpix, group)
    # 4. Return unique (healpix, group) tuples for each dataset
    pass
```

## Testing Strategy

### 1. Test Index Loading

```python
# Use standard DatasetBuilder interface
builder = MMUDatasetBuilder(
    cache_dir=None,
    data_dir="TobiasPitters/mmu-sdss-partitioned",
    split_name="train",
    index_partition="_index"
)

# This should load the index
from datasets.download.download_manager import DownloadManager
dl_manager = DownloadManager()
index = builder._download_and_load_index(dl_manager)
assert len(index) > 0
assert 'ra' in index.column_names
```

### 2. Test Partition Listing

```python
# Should find all healpix partitions
partitions = builder._get_all_partitions(index)
assert len(partitions) > 0
assert all(isinstance(p, tuple) and len(p) == 2 for p in partitions)
```

### 3. Test Full Loading

```python
builder = MMUDatasetBuilder(
    cache_dir=None,
    data_dir="TobiasPitters/mmu-sdss-partitioned",
    split_name="train",
    index_partition="_index"
)

builder.download_and_prepare()
dataset = builder.as_dataset()
assert len(dataset) > 0
```

### 4. Test Crossmatching

```python
from mmu_datasets_ai_slop.matching import spatial_crossmatch_fn

builder = MMUDatasetBuilder(
    cache_dir=None,
    data_dir="TobiasPitters/mmu-sdss-partitioned",
    split_name="train",
    index_partition="_index",
    matching_datasets={'hsc': 'TobiasPitters/mmu-hsc-partitioned'},
    matching_fn=spatial_crossmatch_fn,
    matching_config={'tolerance': 1.0}
)

builder.download_and_prepare()
dataset = builder.as_dataset()
# Should only contain matched objects
```

## Common Issues

### Issue 1: "Couldn't find cache"

**Problem**: HuggingFace datasets caching gets confused.

**Solution**: Clear cache or use `download_mode='force_redownload'`

```python
builder.download_and_prepare(download_mode='force_redownload')
```

### Issue 2: Schema mismatch

**Problem**: Index and data partitions have different schemas.

**Solution**: This is expected! Only load index in `_split_generators()`, then load data separately.

### Issue 3: URLs not found

**Problem**: `hf://` URLs not resolving.

**Solution**: Make sure repo_id is correct and files exist. Test with:

```python
from huggingface_hub import HfFileSystem
fs = HfFileSystem()
files = fs.ls("datasets/TobiasPitters/mmu-sdss-partitioned/train/_index")
```

## Reference Implementation

See `mmu_datasets_ai_slop/` for a full reference implementation (may be overcomplicated, but shows one way to do it).

## Next Steps

1. Start with simple case: load a single dataset without crossmatching
2. Add partition filtering based on index
3. Implement crossmatching logic
4. Test with real SDSS/HSC datasets
5. Optimize and refine

Good luck! 🚀
