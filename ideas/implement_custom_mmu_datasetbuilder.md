# Custom MMU DatasetBuilder Implementation

## Overview

This implementation enables efficient crossmatching of large astronomical datasets by using a two-tier partitioning strategy:

**Partitioning Strategy:**
- Data is partitioned by two keys: `healpix` (spatial partitioning) and `object_group_id` (grouping multiple object IDs together for efficient loading)
- A special `_index` partition is created that contains only the essential coordinate information: `healpix`, `ra`, `dec`, `object_group_id`, and `object_id`
- This `_index` partition is lightweight and can be loaded entirely into memory

**Directory Structure Example:**
```
dataset/
├── _index/                           # Lightweight index partition
│   ├── index_000.parquet            # Contains: healpix, ra, dec, object_group_id, object_id
│   ├── index_001.parquet
│   └── ...
├── healpix=0/                        # Spatial partition (HEALPix cell 0)
│   ├── object_group_id=0/           # Object group 0
│   │   └── data.parquet             # Full data for these objects
│   ├── object_group_id=1/           # Object group 1
│   │   └── data.parquet
│   └── ...
├── healpix=1/                        # Spatial partition (HEALPix cell 1)
│   ├── object_group_id=0/
│   │   └── data.parquet
│   └── ...
└── healpix=2/
    └── ...
```

**Crossmatching Workflow:**
1. Load the `_index` partition for both datasets (small, fast)
2. Perform spatial crossmatching on the loaded indices
3. Identify which `(healpix, object_group_id)` partitions contain successfully crossmatched objects
4. Download and load only those specific data partitions that contain matched objects
5. Skip downloading partitions that don't contain any matched objects

This should dramatically reduce time and memory overhead.

---

## Disclaimer

**Note:** The code below was generated using Claude as a rough architectural sketch. This is an exploratory idea and has not been tested or validated. The implementation details may be incomplete, incorrect, or impractical. Treat this as a starting point for discussion. This could very well be AI slop.

---

# mmu_datasets/builder.py

```python
from datasets import DatasetBuilder, BuilderConfig, Features, Split, SplitGenerator
from datasets.download import DownloadManager
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional, Any


class MMUConfig(BuilderConfig):
    """Configuration for MMU datasets with crossmatching support"""

    def __init__(
        self,
        matching_datasets: Optional[Dict[str, str]] = None,
        matching_fn: Optional[Callable] = None,
        matching_config: Optional[Dict[str, Any]] = None,
        index_partition: str = "_index",
        **kwargs
    ):
        """
        Initialize MMU configuration.

        Args:
            matching_datasets: Dict mapping {name: dataset_path} for datasets to crossmatch with
            matching_fn: Function(primary_index, other_indices, config) -> List[(healpix, group)]
            matching_config: Configuration dict passed to matching_fn (e.g., {"tolerance": 1.0})
            index_partition: Name of the index partition directory (default: "_index")
            **kwargs: Additional BuilderConfig arguments
        """
        pass


class MMUDatasetBuilder(DatasetBuilder):
    """
    Custom DatasetBuilder for Multimodal Universe datasets.

    Implements efficient crossmatching by:
    1. Loading _index partition first
    2. Applying crossmatch function to filter partitions
    3. Downloading only relevant data partitions
    """
    
    BUILDER_CONFIG_CLASS = MMUConfig
    
    def __init__(self, *args, **kwargs):
        """Initialize builder and prepare for index-aware loading."""
        pass
    
    def _info(self) -> datasets.DatasetInfo:
        """
        Define dataset metadata and schema.
        
        Returns:
            DatasetInfo with features, description, etc.
        """
        pass
    
    def download_and_prepare(
        self,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[GenerateMode] = None,
        **kwargs
    ):
        """
        Override download_and_prepare to implement custom logic:
        1. Download _index partition first
        2. Apply matching function if configured
        3. Determine relevant partitions
        4. Download only those partitions
        
        Args:
            download_config: Configuration for downloading
            download_mode: Mode for generation (reuse cache, force redownload, etc.)
            **kwargs: Additional arguments
        """
        pass
    
    def _get_or_create_download_manager(
        self,
        download_config: Optional[DownloadConfig] = None,
        **kwargs
    ) -> DownloadManager:
        """
        Get existing or create new DownloadManager.
        
        Args:
            download_config: Configuration for downloading
            **kwargs: Additional arguments
            
        Returns:
            DownloadManager instance
        """
        pass
    
    def _download_and_load_index(self, dl_manager: DownloadManager) -> pa.Table:
        """
        Download and load the _index partition into memory.
        
        Args:
            dl_manager: DownloadManager for handling downloads
            
        Returns:
            PyArrow Table containing the index data
        """
        pass
    
    def _get_index_urls(self) -> List[str]:
        """
        Get URLs/paths for all files in the _index partition.
        
        Uses HfFileSystem to list files in the repository and filter
        for those in the index partition.
        
        Returns:
            List of HF URLs (e.g., ["hf://datasets/org/dataset/_index/file.parquet"])
        """
        pass
    
    def _list_repository_files(self) -> List[str]:
        """
        List all files in the dataset repository.
        
        Returns:
            List of file paths in the repository
        """
        pass
    
    def _apply_crossmatch(self, dl_manager: DownloadManager) -> List[Tuple[int, int]]:
        """
        Apply crossmatch function to determine relevant partitions.
        
        Steps:
        1. Load indices from other datasets
        2. Call user's matching_fn with all indices
        3. Get list of (healpix, group) tuples to load
        
        Args:
            dl_manager: DownloadManager for downloading other indices
            
        Returns:
            List of (healpix, group) tuples identifying partitions to load
        """
        pass
    
    def _load_other_dataset_indices(
        self,
        dl_manager: DownloadManager
    ) -> Dict[str, pa.Table]:
        """
        Load index partitions from other datasets specified in config.
        
        Args:
            dl_manager: DownloadManager for downloading
            
        Returns:
            Dict mapping dataset name to its index PyArrow Table
        """
        pass
    
    def _get_all_partitions(self) -> List[Tuple[int, int]]:
        """
        Get all unique (healpix, group) partitions from the index.
        
        Used when no matching function is specified - loads everything.
        
        Returns:
            List of all (healpix, group) tuples in the dataset
        """
        pass
    
    def _extract_unique_partitions(self, index_table: pa.Table) -> List[Tuple[int, int]]:
        """
        Extract unique (healpix, group) combinations from index table.
        
        Args:
            index_table: PyArrow table with 'healpix' and 'object_id_group' columns
            
        Returns:
            List of unique (healpix, group) tuples
        """
        pass
    
    def _split_generators(self, dl_manager: DownloadManager) -> List[SplitGenerator]:
        """
        Generate dataset splits based on filtered partitions.
        
        Steps:
        1. Build list of data files from relevant partitions
        2. Download those files
        3. Return SplitGenerator with downloaded files
        
        Args:
            dl_manager: DownloadManager for downloading
            
        Returns:
            List of SplitGenerator objects
        """
        pass
    
    def _build_data_files_list(
        self,
        partitions: List[Tuple[int, int]],
        all_files: List[str]
    ) -> List[str]:
        """
        Build list of data file URLs for specified partitions.
        
        Args:
            partitions: List of (healpix, group) tuples
            all_files: All files in the repository
            
        Returns:
            List of HF URLs for data files in specified partitions
        """
        pass
    
    def _file_matches_partition(
        self,
        file_path: str,
        healpix: int,
        group: int
    ) -> bool:
        """
        Check if a file path matches a partition specification.
        
        Args:
            file_path: File path to check
            healpix: HEALPix value
            group: Object ID group value
            
        Returns:
            True if file belongs to the partition
        """
        pass
    
    def _generate_examples(
        self,
        files: List[str],
        index: pa.Table
    ):
        """
        Generate examples from downloaded parquet files.
        
        Args:
            files: List of downloaded file paths
            index: Index table (for reference/metadata)
            
        Yields:
            Tuple of (key, example_dict)
        """
        pass
    
    def _read_parquet_file(self, file_path: str) -> pa.Table:
        """
        Read a single parquet file into PyArrow Table.
        
        Args:
            file_path: Path to parquet file
            
        Returns:
            PyArrow Table
        """
        pass
    
    def _table_to_examples(self, table: pa.Table):
        """
        Convert PyArrow Table to example dictionaries.
        
        Args:
            table: PyArrow Table
            
        Yields:
            Tuple of (object_id, example_dict)
        """
        pass
```

# mmu_datasets/loader.py

```python
def load_dataset(
    dataset_path: str,
    cross_match: Optional[Tuple[str, ...]] = None,
    tolerance: float = 1.0,
    streaming: bool = False,
    **kwargs
) -> Union[Dataset, IterableDataset, CrossMatchedDatasets]:
    """
    Load MMU dataset with optional crossmatching.
    
    Examples:
        # Simple loading
        >>> ds = load_dataset("MultimodalUniverse/jwst")
        
        # Crossmatching two surveys
        >>> ds = load_dataset(
        ...     "MultimodalUniverse",
        ...     cross_match=("jwst", "hsc"),
        ...     tolerance=1.0,
        ...     streaming=True
        ... )
    
    Args:
        dataset_path: Path to dataset or base path if cross_match is used
        cross_match: Tuple of survey names to crossmatch (e.g., ("jwst", "hsc"))
        tolerance: Matching tolerance in arcseconds
        streaming: Whether to return streaming dataset
        **kwargs: Additional arguments passed to builder
        
    Returns:
        Dataset, IterableDataset, or CrossMatchedDatasets if multiple surveys
    """
    pass


def _load_single_dataset(
    dataset_path: str,
    config: MMUConfig,
    streaming: bool,
    **kwargs
):
    """
    Load a single dataset without crossmatching.
    
    Args:
        dataset_path: Full path to dataset
        config: MMU configuration
        streaming: Whether to stream
        **kwargs: Additional builder arguments
        
    Returns:
        Dataset or IterableDataset
    """
    pass


def _load_crossmatched_datasets(
    base_path: str,
    surveys: Tuple[str, ...],
    tolerance: float,
    streaming: bool,
    **kwargs
) -> CrossMatchedDatasets:
    """
    Load multiple surveys with crossmatching applied.
    
    Args:
        base_path: Base path (e.g., "MultimodalUniverse")
        surveys: Tuple of survey names (e.g., ("jwst", "hsc"))
        tolerance: Matching tolerance in arcseconds
        streaming: Whether to stream
        **kwargs: Additional builder arguments
        
    Returns:
        CrossMatchedDatasets containing aligned datasets
    """
    pass


def _build_crossmatch_config(
    primary_survey: str,
    other_surveys: List[str],
    base_path: str,
    tolerance: float
) -> MMUConfig:
    """
    Build configuration for crossmatching.
    
    Args:
        primary_survey: Primary survey name
        other_surveys: Other survey names to match with
        base_path: Base dataset path
        tolerance: Matching tolerance in arcseconds
        
    Returns:
        MMUConfig with matching configuration
    """
    pass


class CrossMatchedDatasets:
    """Container for multiple crossmatched datasets with aligned iteration."""
    
    def __init__(self, datasets: Dict[str, Union[Dataset, IterableDataset]]):
        """
        Initialize with dict of datasets.
        
        Args:
            datasets: Dict mapping survey name to dataset
        """
        pass
    
    def __iter__(self):
        """
        Iterate over matched examples from all datasets simultaneously.
        
        Yields:
            Dict mapping survey name to example dict
        """
        pass
    
    def __getitem__(self, key: str):
        """
        Access individual dataset by survey name.
        
        Args:
            key: Survey name
            
        Returns:
            Dataset for that survey
        """
        pass
    
    def __len__(self) -> int:
        """
        Get number of matched examples (assumes all datasets same length).
        
        Returns:
            Number of examples
        """
        pass
    
    def _create_aligned_iterators(self) -> Dict[str, Iterator]:
        """
        Create iterators for all datasets.
        
        Returns:
            Dict mapping survey name to iterator
        """
        pass
    
    def map(self, function: Callable, **kwargs):
        """
        Apply function to all datasets.
        
        Args:
            function: Function to apply to examples
            **kwargs: Additional map arguments
            
        Returns:
            New CrossMatchedDatasets with mapped datasets
        """
        pass
    
    def filter(self, function: Callable, **kwargs):
        """
        Filter all datasets (maintains alignment).
        
        Args:
            function: Filter function
            **kwargs: Additional filter arguments
            
        Returns:
            New CrossMatchedDatasets with filtered datasets
        """
        pass
```

# mmu_datasets/matching.py

```python
def spatial_crossmatch_fn(
    primary_index: pa.Table,
    other_indices: Dict[str, pa.Table],
    config: Dict[str, Any]
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Crossmatch astronomical catalogs by spatial position.
    
    Args:
        primary_index: Index table for primary dataset
        other_indices: Dict of {survey_name: index_table}
        config: Configuration dict with 'tolerance' in arcseconds
        
    Returns:
        Dict mapping dataset name to list of (healpix, group) partitions to load
        Example: {
            'primary': [(0, 0), (0, 1), (1, 0)],
            'hsc': [(0, 0), (1, 0), (2, 1)]
        }
    """
    pass


def _extract_coordinates(index_table: pa.Table) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract RA/Dec coordinates from index table.
    
    Args:
        index_table: Index table with 'ra' and 'dec' columns
        
    Returns:
        Tuple of (ra_array, dec_array) in degrees
    """
    pass


def _healpix_crossmatch(
    ra1: np.ndarray,
    dec1: np.ndarray,
    hp1: np.ndarray,
    ra2: np.ndarray,
    dec2: np.ndarray,
    hp2: np.ndarray,
    tolerance: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform HEALPix-accelerated crossmatch.
    
    Args:
        ra1, dec1: Coordinates of first catalog (degrees)
        hp1: HEALPix indices of first catalog
        ra2, dec2: Coordinates of second catalog (degrees)
        hp2: HEALPix indices of second catalog
        tolerance: Matching tolerance in degrees
        
    Returns:
        Tuple of (matches_idx1, matches_idx2) - indices of matched objects
    """
    pass


def _find_neighbor_healpix(healpix: int, nside: int) -> List[int]:
    """
    Find neighboring HEALPix cells (for boundary cases).
    
    Args:
        healpix: HEALPix index
        nside: HEALPix nside parameter
        
    Returns:
        List of neighboring HEALPix indices (including self)
    """
    pass


def _angular_distance(
    ra1: np.ndarray,
    dec1: np.ndarray,
    ra2: np.ndarray,
    dec2: np.ndarray
) -> np.ndarray:
    """
    Calculate angular distance between coordinates using haversine formula.
    
    Args:
        ra1, dec1: First set of coordinates (degrees)
        ra2, dec2: Second set of coordinates (degrees)
        
    Returns:
        Angular distances in degrees
    """
    pass


def _group_by_partition(
    matched_indices: np.ndarray,
    healpix: np.ndarray,
    groups: np.ndarray
) -> List[Tuple[int, int]]:
    """
    Group matched indices by their (healpix, group) partition.
    
    Args:
        matched_indices: Indices of matched objects
        healpix: HEALPix values for all objects
        groups: Object ID group values for all objects
        
    Returns:
        List of unique (healpix, group) tuples
    """
    pass
```

# mmu_datasets/utils.py

```python
def validate_index_schema(index_table: pa.Table):
    """
    Validate that index table has required columns.
    
    Required columns:
    - object_id: string
    - ra: float32
    - dec: float32
    - healpix: int32
    - object_id_group: int16
    
    Args:
        index_table: Index table to validate
        
    Raises:
        ValueError: If schema is invalid
    """
    pass


def estimate_memory_usage(n_objects: int) -> Dict[str, str]:
    """
    Estimate memory usage for index and matching.
    
    Args:
        n_objects: Number of objects in catalog
        
    Returns:
        Dict with memory estimates for different operations
    """
    pass


def create_index_from_catalog(
    catalog_path: str,
    output_path: str,
    nside: int = 2048,
    objects_per_group: int = 10000
):
    """
    Create _index partition from full catalog.
    
    Args:
        catalog_path: Path to full catalog parquet files
        output_path: Where to write _index partition
        nside: HEALPix nside parameter
        objects_per_group: Objects per group for partitioning
    """
    pass


def _compute_healpix(ra: np.ndarray, dec: np.ndarray, nside: int) -> np.ndarray:
    """
    Compute HEALPix indices for coordinates.
    
    Args:
        ra, dec: Coordinates in degrees
        nside: HEALPix nside parameter
        
    Returns:
        HEALPix indices
    """
    pass


def _assign_groups(object_ids: np.ndarray, objects_per_group: int) -> np.ndarray:
    """
    Assign objects to groups for partitioning.

    Args:
        object_ids: Array of object IDs
        objects_per_group: Target objects per group

    Returns:
        Array of group assignments
    """
    pass
```
