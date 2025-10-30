import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from datasets import load_dataset
from datasets.download.download_manager import DownloadManager



# Configuration
REPO_ID = "TobiasPitters/mmu-sdss-partitioned"
FILE_PATH = "train/healpix=1172/data.parquet"  # Example file


def demo_load_single_col():
    hf_fs = HfFileSystem()
    full_path = f"datasets/{REPO_ID}/{FILE_PATH}"
    columns_to_load = ['object_id']
    table = pq.read_table(
        hf_fs.open(full_path, "rb"),
        columns=columns_to_load
    )
    return table

def demo_load_full_file():
    # Use standard datasets loading
    ds = load_dataset(
        "parquet",
        data_files=f"hf://datasets/{REPO_ID}/{FILE_PATH}",
        download_mode="force_redownload"
    )
    return ds['train']

def demo_2_find_row_group():
    hf_fs = HfFileSystem()
    full_path = f"datasets/{REPO_ID}/{FILE_PATH}"

    # Open parquet file
    parquet_file = pq.ParquetFile(hf_fs.open(full_path, "rb"))

    print(f"File has {parquet_file.num_row_groups} row groups")
    print()

    object_id_table = pq.read_table(
        hf_fs.open(full_path, "rb"),
        columns=['object_id']
    )

    target_object_id = object_id_table['object_id'][-1].as_py()

    import pyarrow.compute as pc
    matches = pc.equal(object_id_table['object_id'], target_object_id)
    row_index = pc.index(matches, True).as_py()

    cumulative_rows = 0
    target_row_group = None

    for i in range(parquet_file.num_row_groups):
        row_group = parquet_file.metadata.row_group(i)
        rows_in_group = row_group.num_rows

        if cumulative_rows <= row_index < cumulative_rows + rows_in_group:
            target_row_group = i
            # row_offset_in_group = row_index - cumulative_rows
            break

        cumulative_rows += rows_in_group

    return target_row_group, target_object_id


if __name__ == "__main__":
    import time
    import sys
    # Benchmark 1: Load single column
    print("Test 1: Loading single column (object_id)")
    print("-" * 60)
    start_time = time.time()
    table = demo_load_single_col()
    elapsed_1 = time.time() - start_time

    print(f"✓ Loaded {len(table):,} rows")
    print(f"✓ Columns: {table.column_names}")
    print(f"✓ Memory size: {table.nbytes / 1024 / 1024:.2f} MB")
    print(f"✓ Time: {elapsed_1:.2f} seconds")
    print()

    # Benchmark 2: Load full file
    print("Test 2: Loading full file (all columns)")
    print("-" * 60)
    start_time = time.time()
    full_dataset = demo_load_full_file()
    elapsed_2 = time.time() - start_time

    print(f"✓ Loaded {len(full_dataset):,} rows")
    print(f"✓ Columns: {full_dataset.column_names}")
    print(f"✓ Memory size: {full_dataset.data.nbytes / 1024 / 1024:.2f} MB")
    print(f"✓ Time: {elapsed_2:.2f} seconds")
    print()

    # Comparison
    print("=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"Time difference: {elapsed_2 / elapsed_1:.1f}x slower for full load")
    print(f"Speedup: {((elapsed_2 - elapsed_1) / elapsed_2 * 100):.1f}% faster with partial load")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Single column load: {elapsed_1:.2f}s")
    print("Recommendation: Use partial loading (pq.read_table with columns)")
    print("for targeted queries to minimize bandwidth and time.")

