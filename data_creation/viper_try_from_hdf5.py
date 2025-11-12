import h5py
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import numpy as np

storage_path = Path("data/MultimodalUniverse/v1/vipers/vipers_w1/")
target_parquet_path = Path("data/parquet/vipers_w1/")
chunk_size = 10000
_FLOAT_FEATURES = [
    'REDSHIFT',
    'REDFLAG',
    'EXPTIME',
    'NORM',
    'MAG',
    'ra',
    'dec',
]

def infer_schema_from_hdf5(f):
    """Infer PyArrow schema from HDF5 file."""
    fields = []

    for key in f.keys():
        if isinstance(f[key], h5py.Dataset):
            dataset = f[key]
            dtype = dataset.dtype
            shape = dataset.shape

            # Map numpy dtype to PyArrow type
            if dtype == np.float32:
                pa_type = pa.float32()
            elif dtype == np.float64:
                pa_type = pa.float64()
            elif dtype == np.int32:
                pa_type = pa.int32()
            elif dtype == np.int64:
                pa_type = pa.int64()
            else:
                pa_type = pa.from_numpy_dtype(dtype)

            # Handle multidimensional arrays as lists
            if len(shape) > 1:
                pa_type = pa.list_(pa_type)

            fields.append(pa.field(key, pa_type))

    return pa.schema(fields)

def transform_func(batch: pa.Table):
    spectrum_col = pa.StructArray.from_arrays(
          [
              batch["spectrum_flux"],
              batch["spectrum_noise"],
              batch["spectrum_wave"],
              batch["spectrum_mask"]
          ],
          names=["flux", "ivar", "lambda", "mask"]
    )
    batch = batch.append_column("spectrum", spectrum_col)

    return batch, batch.schema



for healpix in storage_path.iterdir():
    for file in healpix.iterdir():
        with h5py.File(file, 'r') as f:
            base_schema = infer_schema_from_hdf5(f)
            print(f"Base schema: {base_schema}")

            total_rows = f[list(f.keys())[0]].shape[0]
            print(f"Total rows to process: {total_rows}")

            # Run transform once to get final schema
            print("\nInferring final schema by running transform on first row...")
            sample_data = {key: f[key][0:1] for key in f.keys()}
            sample_arrays = []
            for field in base_schema:
                arr = sample_data[field.name]
                if arr.ndim > 1:
                    arr = pa.array(list(arr))
                else:
                    arr = pa.array(arr)
                sample_arrays.append(arr)
            sample_batch = pa.RecordBatch.from_arrays(sample_arrays, schema=base_schema)
            _, final_schema = transform_func(sample_batch)
            print(f"Final schema: {final_schema}")

            # Now write with final schema
            with pq.ParquetWriter('output.parquet', final_schema) as writer:
                for i in range(0, total_rows, chunk_size):
                    print(f"\nProcessing chunk starting at row {i}")
                    data = {key: f[key][i:i+chunk_size] for key in f.keys()}

                    # Convert numpy arrays to PyArrow arrays, handling multidimensional data
                    arrays = []
                    for field in base_schema:
                        arr = data[field.name]
                        if arr.ndim > 1:
                            arr = pa.array(list(arr))
                        else:
                            arr = pa.array(arr)
                        arrays.append(arr)

                    batch = pa.RecordBatch.from_arrays(arrays, schema=base_schema)
                    batch, _ = transform_func(batch)

                    writer.write_batch(batch)
                    print(f"Wrote {batch.num_rows} rows")

                print("\nFinished writing all batches")

