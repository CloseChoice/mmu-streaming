import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from datasets import load_dataset
from datasets.download.download_manager import DownloadManager
import pyarrow.compute as pc



# Configuration
REPO_ID = "MultimodalUniverse/jwst"
FILE_PATH = "all/train-00000-of-00027.parquet"  # Example file

def demo_3_load_specific_row_group(row_group_index: int, target_id: str):
    hf_fs = HfFileSystem()
    full_path = f"datasets/{REPO_ID}/{FILE_PATH}"

    parquet_file = pq.ParquetFile(hf_fs.open(full_path, "rb"))

    table = parquet_file.read_row_group(
        row_group_index,
        columns=['object_id', 'healpix', 'ra', 'dec']  # Can still do column pruning!
    )

    # Verify our target object is in this row group
    matches = pc.equal(table['object_id'], target_id)
    if pc.any(matches).as_py():
        row_idx = pc.index(matches, True).as_py()
        row = table.slice(row_idx, 1).to_pydict()
    else:
        print(f"✗ object_id '{target_id}' not found in this row group")
    return row

def demo_load_full_file(target_id: str):
    hf_fs = HfFileSystem()
    full_path = f"datasets/{REPO_ID}/{FILE_PATH}"

    parquet_file = pq.ParquetFile(hf_fs.open(full_path, "rb"))

    print(f"File has {parquet_file.num_row_groups} row groups")
    print()

    table = pq.read_table(
        hf_fs.open(full_path, "rb"),
        columns=['object_id']
    )
    matches = pc.equal(table['object_id'], target_id)
    if pc.any(matches).as_py():
        row_idx = pc.index(matches, True).as_py()
        row = table.slice(row_idx, 1).to_pydict()
    else:
        print(f"✗ object_id '{target_id}' not found in this row group")
    return row


if __name__ == "__main__":
    row = demo_3_load_specific_row_group(row_group_index=3, target_id="1757963689505762345")
    row2 = demo_load_full_file(target_id="1757963689505762345")
    assert row == row2

