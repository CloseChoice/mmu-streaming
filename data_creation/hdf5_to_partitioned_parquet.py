#!/usr/bin/env python3
# Direct HDF5 to partitioned parquet conversion
# Run with: python data_creation/hdf5_to_partitioned_parquet.py

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
DATASET = "sdss"  # or "hsc"
LOCAL_DATA_DIR = Path(f"scripts/data/MultimodalUniverse/v1/{DATASET}")
REPO_ID = f"TobiasPitters/mmu-{DATASET}-with-coordinates"

# SDSS-specific features
_FLOAT_FEATURES = ["VDISP", "VDISP_ERR", "Z", "Z_ERR"]
_FLUX_FEATURES = ["SPECTROFLUX", "SPECTROFLUX_IVAR", "SPECTROSYNFLUX", "SPECTROSYNFLUX_IVAR"]
_FLUX_FILTERS = ['U', 'G', 'R', 'I', 'Z']
_BOOL_FEATURES = ["ZWARNING"]

# ============================================================================
# Find all HDF5 files
# ============================================================================
hdf5_pattern = "sdss/healpix=*/*.hdf5" if DATASET == "sdss" else "pdr3_dud_22.5/healpix=*/*.hdf5"
hdf5_files = sorted(LOCAL_DATA_DIR.glob(hdf5_pattern))

# ============================================================================
# Group data by healpix
# ============================================================================
healpix_data = defaultdict(list)

for hdf5_file in hdf5_files:
    with h5py.File(hdf5_file, "r") as f:
        num_objects = len(f["object_id"][:])

        for i in range(num_objects):
            # Extract all data for this object
            example = {}

            # Spectrum data
            example["spectrum_flux"] = f["spectrum_flux"][i]
            example["spectrum_ivar"] = f["spectrum_ivar"][i]
            example["spectrum_lsf_sigma"] = f["spectrum_lsf_sigma"][i]
            example["spectrum_lambda"] = f["spectrum_lambda"][i]
            example["spectrum_mask"] = f["spectrum_mask"][i]

            # Float features
            for feat in _FLOAT_FEATURES:
                example[feat] = f[feat][i].astype("float32")

            # Flux features (5 bands each)
            for feat in _FLUX_FEATURES:
                for n, band in enumerate(_FLUX_FILTERS):
                    example[f"{feat}_{band}"] = f[feat][i][n].astype("float32")

            # Bool features
            for feat in _BOOL_FEATURES:
                example[feat] = bool(f[feat][i])

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
        if key.startswith("spectrum_"):
            # Handle spectrum arrays - these are 2D
            arr = np.array(data_dict[key])
            arrays.append(pa.array(list(arr)))
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
        elif key in _BOOL_FEATURES:
            try:
                arrays.append(pa.array(data_dict[key], type=pa.bool_()))
                fields.append(pa.field(key, pa.bool_()))
            except Exception:
                import pdb; pdb.set_trace()
                arrays.append(pa.array(data_dict[key], type=pa.bool_()))
                fields.append(pa.field(key, pa.bool_()))

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
