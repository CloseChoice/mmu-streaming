import h5py
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import numpy as np

storage_path = Path("data/MultimodalUniverse/v1/vipers/vipers_w1/")
target_parquet_path = Path("data/parquet/vipers_w1/")

_FLOAT_FEATURES = [
    'REDSHIFT', 
    'REDFLAG', 
    'EXPTIME', 
    'NORM', 
    'MAG',
    'ra',
    'dec',
]

all_examples = []

for healpix in storage_path.iterdir():
    for file in healpix.iterdir():
        # instead of this, try pq.from_hdf5 and check if it results in the same!
        with h5py.File(file, "r") as data:    
            keys = data["object_id"]

            # Preparing an index for fast searching through the catalog
            sort_index = np.argsort(data["object_id"])
            sorted_ids = data["object_id"][:][sort_index]

            for k in keys:
                # Extract the indices of requested ids in the catalog
                i = sort_index[np.searchsorted(sorted_ids, k)]

                example = {
                    "object_id": k,
                    "spectrum": {
                        "flux": data["spectrum_flux"][i] * 1e17, # normalize
                        "ivar": 1/(data["spectrum_noise"][i] * 1e34), # normalize
                        "lambda": data["spectrum_wave"][i],
                        "mask": data["spectrum_mask"][i]
                    }
                }

                for key in _FLOAT_FEATURES:
                    example[key] = data[key][i].astype(np.float32)

                all_examples.append(example)

# Flatten the nested structure for Arrow/Parquet
flattened_data = {
    'object_id': [],
    'spectrum_flux': [],
    'spectrum_ivar': [],
    'spectrum_lambda': [],
    'spectrum_mask': [],
}

# Add scalar feature columns
for key in _FLOAT_FEATURES:
    flattened_data[key] = []

# Flatten all examples
for example in all_examples:
    flattened_data['object_id'].append(example['object_id'])
    flattened_data['spectrum_flux'].append(example['spectrum']['flux'])
    flattened_data['spectrum_ivar'].append(example['spectrum']['ivar'])
    flattened_data['spectrum_lambda'].append(example['spectrum']['lambda'])
    flattened_data['spectrum_mask'].append(example['spectrum']['mask'])

    for key in _FLOAT_FEATURES:
        flattened_data[key].append(example[key])

# Convert to Arrow table
final_table = pa.table(flattened_data)

# Create output directory if it doesn't exist
target_parquet_path.mkdir(parents=True, exist_ok=True)

# Write to Parquet file
pq.write_table(final_table, target_parquet_path / "vipers_w1.parquet")
