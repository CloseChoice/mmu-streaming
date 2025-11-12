from hats_import.catalog.arguments import ImportArguments
from hats_import.pipeline import pipeline

args = ImportArguments(
    sort_columns="object_id",
    ra_column="ra",
    dec_column="dec",
    input_path="./data/parquet/vipers_w1/",
    file_reader="parquet",
    output_artifact_name="vipers_w1_catalog",
    output_path="./data/hats/vipers_w1/",
)

# Run the hats import pipeline
if __name__ == "__main__":
    pipeline(args)
