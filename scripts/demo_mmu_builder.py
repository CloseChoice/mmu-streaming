"""Demonstration script for using MMUConfig and MMUDatasetBuilder.

This script shows the intended usage pattern for the custom dataset builder.
Use this as a reference while implementing the actual functionality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mmu_datasets.builder import MMUConfig, MMUDatasetBuilder


def demo_simple_loading():
    builder = MMUDatasetBuilder(
        # cache_dir=None,  # Use default cache
        left_dataset="TobiasPitters/mmu-sdss-with-coordinates",
        right_dataset="TobiasPitters/mmu-hsc-with-coordinates",
        # MMU-specific config params (no type annotations in function calls!)
        split_name="train",
        index_partition="_index",
        columns=None,  # Load all columns, or specify: ["ra", "dec", "flux"]
    )

    builder.download_and_prepare()
    dataset = builder.as_dataset()


def demo_crossmatching():
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
        # matching_fn=spatial_crossmatch_fn,
        matching_config={
            "tolerance": 1.0,  # arcseconds
        },
    )



def main():
    # Run demos
    demo_simple_loading()

if __name__ == "__main__":
    main()
