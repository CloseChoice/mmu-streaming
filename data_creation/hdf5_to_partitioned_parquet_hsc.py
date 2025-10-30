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
# Group data by healpix
# ============================================================================
healpix_data = defaultdict(list)

for hdf5_file in hdf5_files:
    with h5py.File(hdf5_file, "r") as f:
        num_objects = len(f["object_id"][:])

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

            # Group by healpix
            healpix_data[example["healpix"]].append(example)

# ============================================================================
# Create output directory structure
# ============================================================================
upload_path = Path(f"output_dataset/{DATASET}_with_coords")
output_dir = upload_path / "train"
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Save each healpix partition as parquet
# ============================================================================
for idx, (hp_value, examples) in enumerate(healpix_data.items()):
    hp_dir = output_dir / f"healpix={hp_value}"
    hp_dir.mkdir(exist_ok=True)

    # Convert list of dicts to dict of lists
    data_dict = defaultdict(list)
    for example in examples:
        for key, value in example.items():
            data_dict[key].append(value)

    # Create PyArrow table
    arrays = []
    fields = []

    for key in data_dict:
        if key == "image_band":
            # List of strings (5 bands per object)
            arrays.append(pa.array(data_dict[key]))
            fields.append(pa.field(key, pa.list_(pa.string())))
        elif key.startswith("image_flux") or key.startswith("image_ivar"):
            # List of 2D arrays (160x160)
            # Need to flatten each 2D array into a list
            flattened = []
            for obj_bands in data_dict[key]:
                flattened.append([arr.flatten().tolist() for arr in obj_bands])
            arrays.append(pa.array(flattened))
            fields.append(pa.field(key, pa.list_(pa.list_(pa.float32()))))
        elif key.startswith("image_mask"):
            # List of 2D bool arrays
            flattened = []
            for obj_bands in data_dict[key]:
                flattened.append([arr.flatten().tolist() for arr in obj_bands])
            arrays.append(pa.array(flattened))
            fields.append(pa.field(key, pa.list_(pa.list_(pa.bool_()))))
        elif key.startswith("image_"):
            # image_psf_fwhm and image_scale - list of floats (5 per object)
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
            # Float features
            arrays.append(pa.array(data_dict[key], type=pa.float32()))
            fields.append(pa.field(key, pa.float32()))

    schema = pa.schema(fields)
    table = pa.Table.from_arrays(arrays, schema=schema)

    # Write parquet file
    pq.write_table(table, hp_dir / "data.parquet")

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
