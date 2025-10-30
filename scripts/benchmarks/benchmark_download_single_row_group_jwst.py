import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from datasets import load_dataset
from datasets.download.download_manager import DownloadManager
import pyarrow.compute as pc



# Configuration
REPO_ID = "MultimodalUniverse/jwst"
FILE_PATH = "all/train-00000-of-00027.parquet"  # Example file

def find_row_group_and_download(target_id: str):
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

    matches = pc.equal(object_id_table['object_id'], target_id)
    row_index = pc.index(matches, True).as_py()

    cumulative_rows = 0
    target_row_group = None

    for i in range(parquet_file.num_row_groups):
        row_group = parquet_file.metadata.row_group(i)
        rows_in_group = row_group.num_rows

        if cumulative_rows <= row_index < cumulative_rows + rows_in_group:
            target_row_group = i
            break

        cumulative_rows += rows_in_group

    table = parquet_file.read_row_group(
        target_row_group,
        # filters=pc.field("object_id") == target_id
    )

    # Verify our target object is in this row group
    matches = pc.equal(table['object_id'], target_id)
    if pc.any(matches).as_py():
        row_idx = pc.index(matches, True).as_py()
        row = table.slice(row_idx, 1).to_pydict()
        pq.write_table(table.slice(row_idx, 1), "temp_partial.parquet")
        return row
    else:
        print(f"✗ object_id '{target_id}' not found in this row group")


def demo_load_full_file(target_id: str):
    hf_fs = HfFileSystem()
    full_path = f"datasets/{REPO_ID}/{FILE_PATH}"

    parquet_file = pq.ParquetFile(hf_fs.open(full_path, "rb"))

    print(f"File has {parquet_file.num_row_groups} row groups")
    print()

    table = pq.read_table(
        hf_fs.open(full_path, "rb"),
        filters=pc.field("object_id") == target_id
    )
    matches = pc.equal(table['object_id'], target_id)
    if pc.any(matches).as_py():
        row_idx = pc.index(matches, True).as_py()
        row = table.slice(row_idx, 1).to_pydict()
        pq.write_table(table.slice(row_idx, 1), "temp_full.parquet")
        return row
    else:
        print(f"✗ object_id '{target_id}' not found in this row group")


if __name__ == "__main__":
    import time
    target_id = "1757963689505762345"
    t1 = time.time()
    row = find_row_group_and_download(target_id)
    elapsed1 = time.time() - t1
    t2 = time.time()
    row2 = demo_load_full_file(target_id)
    elapsed2 = time.time() - t2
    print(f"Row group: {elapsed1:.2f}s | Full file: {elapsed2:.2f}s | Speedup: {elapsed2/elapsed1:.1f}x")
    import pdb; pdb.set_trace()
    assert row == row2

