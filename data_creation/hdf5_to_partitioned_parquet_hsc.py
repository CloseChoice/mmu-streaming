#!/usr/bin/env python3
# Direct HDF5 to partitioned parquet conversion for HSC
# Run with: python data_creation/hdf5_to_partitioned_parquet_hsc.py

import h5py
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from huggingface_hub import create_repo, HfApi
from collections import defaultdict
import numpy as np

# ============================================================================
# Configuration
# ============================================================================
DATASET = "hsc"
LOCAL_DATA_DIR = Path(f"scripts/data/MultimodalUniverse/v1/{DATASET}")
REPO_ID = f"TobiasPitters/mmu-{DATASET}-with-coordinates"
EXAMPLES_PER_CHUNK = 1000  # Number of examples to accumulate before saving

# HSC-specific features
_FLOAT_FEATURES = [
    'a_g', 'a_r', 'a_i', 'a_z', 'a_y',
    'g_extendedness_value', 'r_extendedness_value', 'i_extendedness_value',
    'z_extendedness_value', 'y_extendedness_value',
    'g_cmodel_mag', 'g_cmodel_magerr',
    'r_cmodel_mag', 'r_cmodel_magerr',
    'i_cmodel_mag', 'i_cmodel_magerr',
    'z_cmodel_mag', 'z_cmodel_magerr',
    'y_cmodel_mag', 'y_cmodel_magerr',
    'g_sdssshape_psf_shape11', 'g_sdssshape_psf_shape22', 'g_sdssshape_psf_shape12',
    'r_sdssshape_psf_shape11', 'r_sdssshape_psf_shape22', 'r_sdssshape_psf_shape12',
    'i_sdssshape_psf_shape11', 'i_sdssshape_psf_shape22', 'i_sdssshape_psf_shape12',
    'z_sdssshape_psf_shape11', 'z_sdssshape_psf_shape22', 'z_sdssshape_psf_shape12',
    'y_sdssshape_psf_shape11', 'y_sdssshape_psf_shape22', 'y_sdssshape_psf_shape12',
    'g_sdssshape_shape11', 'g_sdssshape_shape22', 'g_sdssshape_shape12',
    'r_sdssshape_shape11', 'r_sdssshape_shape22', 'r_sdssshape_shape12',
    'i_sdssshape_shape11', 'i_sdssshape_shape22', 'i_sdssshape_shape12',
    'z_sdssshape_shape11', 'z_sdssshape_shape22', 'z_sdssshape_shape12',
    'y_sdssshape_shape11', 'y_sdssshape_shape22', 'y_sdssshape_shape12'
]

_BANDS = ['G', 'R', 'I', 'Z', 'Y']
_IMAGE_SIZE = 160

# ============================================================================
# Find all HDF5 files
# ============================================================================
hdf5_pattern = "pdr3_dud_22.5/healpix=*/*.hdf5"
hdf5_files = sorted(LOCAL_DATA_DIR.glob(hdf5_pattern))

# ============================================================================
# Helper function to convert examples to PyArrow table
# ============================================================================
def examples_to_table(examples):
    """Convert list of example dicts to PyArrow table"""
    data_dict = defaultdict(list)
    for example in examples:
        for key, value in example.items():
            data_dict[key].append(value)

    arrays = []
    fields = []

    for key in data_dict:
        if key == "image_band":
            arrays.append(pa.array(data_dict[key]))
            fields.append(pa.field(key, pa.list_(pa.string())))
        elif key.startswith("image_flux") or key.startswith("image_ivar"):
            flattened = []
            for obj_bands in data_dict[key]:
                flattened.append([arr.flatten().tolist() for arr in obj_bands])
            arrays.append(pa.array(flattened))
            fields.append(pa.field(key, pa.list_(pa.list_(pa.float32()))))
        elif key.startswith("image_mask"):
            flattened = []
            for obj_bands in data_dict[key]:
                flattened.append([arr.flatten().tolist() for arr in obj_bands])
            arrays.append(pa.array(flattened))
            fields.append(pa.field(key, pa.list_(pa.list_(pa.bool_()))))
        elif key.startswith("image_"):
            arrays.append(pa.array(data_dict[key]))
            fields.append(pa.field(key, pa.list_(pa.float32())))
        elif key == "object_id":
            arrays.append(pa.array(data_dict[key], type=pa.string()))
            fields.append(pa.field(key, pa.string()))
        elif key in ["ra", "dec"]:
            arrays.append(pa.array(data_dict[key], type=pa.float64()))
            fields.append(pa.field(key, pa.float64()))
        elif key == "healpix":
            arrays.append(pa.array(data_dict[key], type=pa.int64()))
            fields.append(pa.field(key, pa.int64()))
        else:
            arrays.append(pa.array(data_dict[key], type=pa.float32()))
            fields.append(pa.field(key, pa.float32()))

    schema = pa.schema(fields)
    return pa.Table.from_arrays(arrays, schema=schema)

# ============================================================================
# Create output directory structure
# ============================================================================
upload_path = Path(f"output_dataset/{DATASET}_with_coords")
output_dir = upload_path / "train"
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Process data in chunks by healpix
# ============================================================================
# Track how many chunks per healpix and current buffer
healpix_chunk_counts = defaultdict(int)
healpix_buffers = defaultdict(list)

for hdf5_file in hdf5_files:
    print(f"Processing {hdf5_file}...")
    with h5py.File(hdf5_file, "r") as f:
        num_objects = len(f["object_id"][:])
        print(f"  Found {num_objects} objects")

        for i in range(num_objects):
            example = {}

            # Image data - structure as list of dicts for each band
            example["image_band"] = []
            example["image_flux"] = []
            example["image_ivar"] = []
            example["image_mask"] = []
            example["image_psf_fwhm"] = []
            example["image_scale"] = []

            for j in range(len(_BANDS)):
                band = f["image_band"][i][j]
                if isinstance(band, bytes):
                    band = band.decode('utf-8')
                example["image_band"].append(band)
                example["image_flux"].append(f["image_array"][i][j])
                example["image_ivar"].append(f["image_ivar"][i][j])
                example["image_mask"].append(f["image_mask"][i][j])
                example["image_psf_fwhm"].append(f["image_psf_fwhm"][i][j])
                example["image_scale"].append(f["image_scale"][i][j])

            # Float features
            for feat in _FLOAT_FEATURES:
                example[feat] = f[feat][i].astype("float32")

            # Object ID and coordinates
            obj_id = f["object_id"][i]
            if isinstance(obj_id, bytes):
                obj_id = obj_id.decode('utf-8')
            else:
                obj_id = str(obj_id)

            example["object_id"] = obj_id
            example["ra"] = f["ra"][i]
            example["dec"] = f["dec"][i]
            example["healpix"] = f["healpix"][i]

            # Add to buffer
            hp_value = example["healpix"]
            healpix_buffers[hp_value].append(example)

            # Save chunk if buffer is full
            if len(healpix_buffers[hp_value]) >= EXAMPLES_PER_CHUNK:
                hp_dir = output_dir / f"healpix={hp_value}"
                hp_dir.mkdir(exist_ok=True)

                chunk_idx = healpix_chunk_counts[hp_value]
                filename = f"train-{chunk_idx:05d}.parquet"

                table = examples_to_table(healpix_buffers[hp_value])
                pq.write_table(table, hp_dir / filename)

                print(f"  Saved {filename} with {len(healpix_buffers[hp_value])} examples")

                # Clear buffer and increment chunk count
                healpix_buffers[hp_value] = []
                healpix_chunk_counts[hp_value] += 1

# ============================================================================
# Save remaining data in buffers
# ============================================================================
print("\nSaving remaining buffered data...")
for hp_value, examples in healpix_buffers.items():
    if len(examples) > 0:
        hp_dir = output_dir / f"healpix={hp_value}"
        hp_dir.mkdir(exist_ok=True)

        chunk_idx = healpix_chunk_counts[hp_value]
        filename = f"train-{chunk_idx:05d}.parquet"

        table = examples_to_table(examples)
        pq.write_table(table, hp_dir / filename)

        print(f"  Saved {filename} with {len(examples)} examples")
        healpix_chunk_counts[hp_value] += 1

# ============================================================================
# Rename files to include total count
# ============================================================================
print("\nRenaming files with total count...")
for hp_value, total_chunks in healpix_chunk_counts.items():
    hp_dir = output_dir / f"healpix={hp_value}"

    for chunk_idx in range(total_chunks):
        old_name = hp_dir / f"train-{chunk_idx:05d}.parquet"
        new_name = hp_dir / f"train-{chunk_idx:05d}-of-{total_chunks:05d}.parquet"

        if old_name.exists():
            old_name.rename(new_name)
            print(f"  Renamed {old_name.name} -> {new_name.name}")

# ============================================================================
# Upload to HuggingFace Hub
# ============================================================================
api = HfApi()

create_repo(
    repo_id=REPO_ID,
    repo_type="dataset",
    exist_ok=True,
)

api.upload_folder(
    folder_path=str(upload_path),
    repo_id=REPO_ID,
    repo_type="dataset",
    path_in_repo=".",
)
