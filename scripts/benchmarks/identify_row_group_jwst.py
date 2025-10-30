import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from datasets import load_dataset
from datasets.download.download_manager import DownloadManager



# Configuration
REPO_ID = "MultimodalUniverse/jwst"
FILE_PATH = "all/train-00000-of-00027.parquet"  # Example file


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
    import pdb; pdb.set_trace()
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
    row_group, object_id = demo_2_find_row_group()
    import pdb; pdb.set_trace()
    print(f"Object ID {object_id} is in row group {row_group}")
