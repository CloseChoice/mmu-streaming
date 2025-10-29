# MMU Datasets - Custom Dataset Builder

Custom HuggingFace dataset builder for Multimodal Universe with efficient crossmatching support.

## Status

🚧 **Under Development** - This is a skeleton/reference implementation. The core functionality needs to be implemented.

## What's Here

### Core Files

- **`builder.py`** - Main implementation file
  - `MMUConfig` - Configuration dataclass (✅ Complete)
  - `MMUDatasetBuilder` - Custom builder (🚧 Skeleton)

- **`__init__.py`** - Package exports (✅ Complete)

- **`IMPLEMENTATION_GUIDE.md`** - Step-by-step implementation guide (📖 Read this!)

### Reference Scripts

- **`scripts/demo_mmu_builder.py`** - Shows intended usage patterns
- **`scripts/test_mmu_builder_minimal.py`** - Minimal test suite to validate implementation

### Reference Implementation

- **`mmu_datasets_ai_slop/`** - Full implementation (may be overcomplicated, but shows one approach)

## Quick Start

### 1. Read the Implementation Guide

```bash
cat mmu_datasets/IMPLEMENTATION_GUIDE.md
```

This explains the architecture and what needs to be implemented.

### 2. Run the Demo Script

```bash
python scripts/demo_mmu_builder.py
```

This shows how the builder should be used (once implemented).

### 3. Run the Minimal Test

```bash
python scripts/test_mmu_builder_minimal.py
```

This tests the parts that are already implemented (config, index loading, etc.)

### 4. Implement the Core Methods

The key methods to implement in `MMUDatasetBuilder`:

1. **`_info()`** - Return dataset metadata
2. **`_split_generators()`** - Main loading logic
3. **`_generate_tables()`** - Yield data from parquet files

See `IMPLEMENTATION_GUIDE.md` for details.

## Dataset Structure

Your HuggingFace dataset should be structured like:

```
repo/
└── train/                     # Split directory
    ├── _index/                # Lightweight index
    │   └── index.parquet     # (object_id, ra, dec, healpix, object_group_id)
    ├── healpix=1172/          # Spatial partition
    │   └── train/
    │       └── data.parquet  # Full dataset
    └── healpix=1173/
        └── train/
            └── data.parquet
```

## Usage (Once Implemented)

### Simple Loading

```python
from mmu_datasets import MMUDatasetBuilder

# Use standard DatasetBuilder interface
builder = MMUDatasetBuilder(
    cache_dir=None,
    data_dir="TobiasPitters/mmu-sdss-partitioned",
    split_name="train",
    index_partition="_index",
)

builder.download_and_prepare()
dataset = builder.as_dataset()
```

### With Crossmatching

```python
from mmu_datasets import MMUDatasetBuilder
from mmu_datasets_ai_slop.matching import spatial_crossmatch_fn

builder = MMUDatasetBuilder(
    cache_dir=None,
    data_dir="TobiasPitters/mmu-sdss-partitioned",
    split_name="train",
    index_partition="_index",
    # Crossmatching config passed as kwargs
    matching_datasets={"hsc": "TobiasPitters/mmu-hsc-partitioned"},
    matching_fn=spatial_crossmatch_fn,
    matching_config={"tolerance": 1.0},
)

builder.download_and_prepare()  # Only downloads matched partitions!
dataset = builder.as_dataset()
```

## Key Concepts

### Index-First Loading

1. Download lightweight `_index` partition first
2. Determine which data partitions are needed
3. Download only those partitions
4. Saves bandwidth and time!

### Partition Filtering

- Data is partitioned by `(healpix, object_group_id)`
- Crossmatching determines which partitions contain matched objects
- Only those partitions are downloaded

### Matching Function

The matching function signature:

```python
def matching_fn(
    primary_index: pa.Table,
    other_indices: Dict[str, pa.Table],
    config: Dict[str, Any]
) -> Dict[str, List[Tuple[int, int]]]:
    # Returns which (healpix, group) partitions to load for each dataset
    pass
```

## Implementation Checklist

- [x] `MMUConfig` dataclass
- [x] `MMUDatasetBuilder` skeleton
- [x] Helper methods (`_get_index_urls`, `_list_repository_files`, etc.)
- [ ] `_info()` method
- [ ] `_split_generators()` method
- [ ] `_generate_tables()` method
- [ ] Crossmatching logic
- [ ] Matching function (`spatial_crossmatch_fn`)
- [ ] Tests
- [ ] Documentation

## Testing Your Implementation

### Step 1: Test Index Loading

```bash
python scripts/test_mmu_builder_minimal.py
```

All tests should pass (config, builder, file listing, index loading).

### Step 2: Test Dataset Loading

Once you implement `_info()`, `_split_generators()`, and `_generate_tables()`:

```python
builder = MMUDatasetBuilder(config=config)
builder.download_and_prepare()
dataset = builder.as_dataset()
assert len(dataset) > 0
```

### Step 3: Test Crossmatching

Implement the matching function and test with two datasets.

## Resources

- **HuggingFace Datasets Docs**: https://huggingface.co/docs/datasets
- **ArrowBasedBuilder Source**: Look at `datasets.packaged_modules.parquet.parquet.Parquet`
- **Reference Implementation**: `mmu_datasets_ai_slop/`

## Getting Help

1. Read `IMPLEMENTATION_GUIDE.md`
2. Check the demo scripts
3. Look at reference implementation in `mmu_datasets_ai_slop/`
4. Review HuggingFace datasets documentation

Good luck! 🚀
