"""Demonstration script for using MMUConfig and MMUDatasetBuilder.

This script shows the intended usage pattern for the custom dataset builder.
Use this as a reference while implementing the actual functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path to import mmu_datasets
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmu_datasets.builder import MMUConfig, MMUDatasetBuilder


def demo_simple_loading():
    """Demo: Load a single dataset without crossmatching."""
    print("=" * 60)
    print("Demo 1: Simple Dataset Loading")
    print("=" * 60)
    print()

    print("Configuration:")
    print("  data_dir: TobiasPitters/mmu-sdss-partitioned")
    print("  split_name: train")
    print("  index_partition: _index")
    print()

    # Create builder - pass config params as kwargs
    print("Creating builder...")
    builder = MMUDatasetBuilder(
        # cache_dir=None,  # Use default cache
        left_dataset="TobiasPitters/mmu-sdss-partitioned",
        right_dataset="TobiasPitters/mmu-hsc-partitioned",
        # MMU-specific config params (no type annotations in function calls!)
        split_name="train",
        index_partition="_index",
        columns=None,  # Load all columns, or specify: ["ra", "dec", "flux"]
    )

    print(f"  Builder class: {type(builder).__name__}")
    print(f"  Config class: {type(builder.config).__name__}")
    print(f"  Config.split_name: {builder.config.split_name}")
    print(f"  Config.index_partition: {builder.config.index_partition}")
    print()

    # The builder should implement these steps:
    print("Expected workflow:")
    print("  1. builder.download_and_prepare()")
    print("     - Downloads and loads _index partition")
    print("     - Determines which data partitions to load")
    print("     - Downloads only needed data partitions")
    print()
    print("  2. dataset = builder.as_dataset()")
    print("     - Returns HuggingFace Dataset with matched objects")
    print()

    # This would actually load the dataset (when implemented)
    try:
        print("Attempting to load dataset...")
        builder.download_and_prepare()
        dataset = builder.as_dataset()
        print(f"✓ Dataset loaded: {len(dataset)} examples")
    except Exception as e:
        print(f"⚠️  Not yet implemented: {e}")

    print()


def demo_crossmatching():
    """Demo: Load two datasets with crossmatching."""
    print("=" * 60)
    print("Demo 2: Crossmatched Dataset Loading")
    print("=" * 60)
    print()

    # Import the matching function (would need to be implemented)
    try:
        from mmu_datasets_ai_slop.matching import spatial_crossmatch_fn
        print("✓ Matching function imported from ai_slop")
    except ImportError:
        print("⚠️  Matching function not found, using None")
        spatial_crossmatch_fn = None

    print(f"Configuration:")
    print(f"  Primary dataset: TobiasPitters/mmu-sdss-partitioned")
    print(f"  Matching with: ['hsc']")
    print(f"  Tolerance: 1.0 arcsec")
    print(f"  Matching function: {spatial_crossmatch_fn}")
    print()

    # Create builder - pass config params as kwargs
    builder = MMUDatasetBuilder(
        cache_dir=None,
        data_dir="TobiasPitters/mmu-sdss-partitioned",
        split_name="train",
        index_partition="_index",
        # Crossmatching configuration
        matching_datasets={
            "hsc": "TobiasPitters/mmu-hsc-partitioned"
        },
        matching_fn=spatial_crossmatch_fn,
        matching_config={
            "tolerance": 1.0,  # arcseconds
        },
    )

    print("Expected workflow:")
    print("  1. builder.download_and_prepare()")
    print("     - Downloads _index for SDSS (primary)")
    print("     - Downloads _index for HSC (other)")
    print("     - Calls matching_fn(sdss_index, {'hsc': hsc_index}, config)")
    print("     - Gets back: {'primary': [(hp1, g1), ...], 'hsc': [(hp2, g2), ...]}")
    print("     - Downloads only matched data partitions")
    print()
    print("  2. dataset = builder.as_dataset()")
    print("     - Returns dataset with only matched SDSS objects")
    print()
    print("  3. Load HSC separately (or return both in a container)")
    print()


def demo_matching_function_signature():
    """Demo: Show what the matching function should look like."""
    print("=" * 60)
    print("Demo 3: Matching Function Signature")
    print("=" * 60)
    print()

    print("The matching function should have this signature:")
    print()
    print("def spatial_crossmatch_fn(")
    print("    primary_index: pa.Table,")
    print("    other_indices: Dict[str, pa.Table],")
    print("    config: Dict[str, Any]")
    print(") -> Dict[str, List[Tuple[int, int]]]:")
    print()
    print("Arguments:")
    print("  - primary_index: PyArrow table with columns:")
    print("      ['object_id', 'ra', 'dec', 'healpix', 'object_group_id']")
    print()
    print("  - other_indices: Dict mapping dataset name to its index table")
    print("      {'hsc': pa.Table(...), 'jwst': pa.Table(...)}")
    print()
    print("  - config: Matching configuration")
    print("      {'tolerance': 1.0}")
    print()
    print("Returns:")
    print("  Dict mapping dataset name to list of (healpix, group) tuples:")
    print("  {")
    print("      'primary': [(1172, 0), (1173, 0), (1173, 1)],  # SDSS partitions")
    print("      'hsc': [(2234, 0), (2234, 1), (2235, 0)]       # HSC partitions")
    print("  }")
    print()
    print("This tells the builder which data partitions to download.")
    print()


def demo_expected_methods():
    """Demo: Show which methods need to be implemented in the builder."""
    print("=" * 60)
    print("Demo 4: Methods to Implement")
    print("=" * 60)
    print()

    print("MMUDatasetBuilder should implement:")
    print()
    print("1. _info() -> DatasetInfo")
    print("   - Returns dataset metadata and schema")
    print()
    print("2. _split_generators(dl_manager) -> List[SplitGenerator]")
    print("   - Main logic:")
    print("     a. Load index partition")
    print("     b. Apply crossmatch (if configured)")
    print("     c. Determine relevant partitions")
    print("     d. Download data files")
    print("     e. Return SplitGenerator with file paths")
    print()
    print("3. _generate_tables(files, index) -> Iterator[Tuple[str, pa.Table]]")
    print("   - Yields tables from downloaded parquet files")
    print()
    print("Helper methods (already partially implemented):")
    print("  - _get_index_urls() -> List[str]")
    print("  - _download_and_load_index(dl_manager) -> pa.Table")
    print("  - _list_repository_files() -> List[str]")
    print("  - _extract_repo_id_from_url(url) -> str")
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("MMU Dataset Builder - Usage Demonstration")
    print("=" * 60)
    print("\nThis script demonstrates the intended usage of MMUConfig")
    print("and MMUDatasetBuilder. Use it as a reference for implementation.")
    print()

    # Run demos
    demo_simple_loading()
    # input("\nPress Enter to continue to next demo...")

    # demo_crossmatching()
    # input("\nPress Enter to continue to next demo...")

    # demo_matching_function_signature()
    # input("\nPress Enter to continue to next demo...")

    # demo_expected_methods()

    print("\n" + "=" * 60)
    print("Key Implementation Steps:")
    print("=" * 60)
    print()
    print("1. Implement _info() to return dataset schema")
    print("2. Implement _split_generators() with:")
    print("   - Index loading")
    print("   - Crossmatch logic")
    print("   - Partition filtering")
    print("   - Data download")
    print("3. Implement _generate_tables() to yield data")
    print("4. Implement matching.py with spatial_crossmatch_fn()")
    print("5. Test with real datasets!")
    print()
    print("See mmu_datasets_ai_slop/ for a full reference implementation")
    print("(though it may be overcomplicated)")
    print()


if __name__ == "__main__":
    main()
