# Prerequisites:
# uv pip install -r requirements.txt
# ./download_sdss_hsc.sh
# NOTE: 
# this also runs with datasets==4 and numpy>1
from datasets import load_dataset
from functions.crossmatch_manual import cross_match_datasets_manual


sdss = load_dataset("TobiasPitters/mmu-sdss-with-coordinates")
hsc = load_dataset("TobiasPitters/mmu-hsc-with-coordinates")

matched = cross_match_datasets_manual(sdss,
                                      hsc,
                                      left_name="sdss",
                                      right_name="hsc",
                                      matching_radius=1.0,
                                      # well, coordinate_columns might not be the best name here
                                      coordinate_columns=['ra', 'dec', 'healpix', 'object_id']
                                      )

assert len(matched) == 25
