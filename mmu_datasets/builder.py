from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from datasets import ArrowBasedBuilder, BuilderConfig, DatasetInfo, Features, SplitGenerator
from datasets.download.download_manager import DownloadManager
from datasets.utils.file_utils import is_remote_url
from huggingface_hub import HfFileSystem
from datasets.packaged_modules.parquet.parquet import Parquet, ParquetConfig
from datasets import config
import numpy as np


from astropy.table import Table, hstack
from astropy.coordinates import SkyCoord
from astropy import units as u

@dataclass
# class MMUConfig(BuilderConfig):
class MMUConfig(ParquetConfig):
    """Configuration for MMU datasets with crossmatching support.

    Attributes:
        name: The name of the configuration.
        version: The version of the configuration.
        data_dir: Path to the directory containing the source data.
        data_files: Path(s) to source data file(s).
        description: A human description of the configuration.
        matching_datasets: Dict mapping {name: dataset_path} for datasets to crossmatch with.
        matching_fn: Function(primary_index, other_indices, config) -> Dict[str, List[Tuple]].
        matching_config: Configuration dict passed to matching_fn (e.g., {"tolerance": 1.0}).
        index_partition: Name of the index partition directory (default: "_index").
        split_name: Name of the split directory (default: "train").
        batch_size: Batch size for reading parquet files.
        columns: Specific columns to load (None means all).
        features: Dataset features schema.
    """

    split_name: str = "train"
    index_partition: str = "_index"
    left_dataset: str = "default_left"
    right_dataset: str = "default_right"
    batch_size: Optional[int] = None
    columns: Optional[List[str]] = None
    features: Optional[Features] = None

    # name: str = "default"
    # version: Optional[Union[utils.Version, str]] = utils.Version("0.0.0")
    # data_dir: Optional[str] = None
    # data_files: Optional[Union[DataFilesDict, DataFilesPatternsDict]] = None
    # description: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()

class MMUDatasetBuilder(Parquet):
    """Custom DatasetBuilder for Multimodal Universe datasets.

    Implements efficient crossmatching by:
    1. Loading _index partition first
    2. Applying crossmatch function to filter partitions
    3. Downloading only relevant data partitions
    """

    BUILDER_CONFIG_CLASS = MMUConfig

    def __init__(self, *args, **kwargs):
        """Initialize builder and prepare for index-aware loading."""
        left_dataset = kwargs.get("left_dataset", "default_left")
        right_dataset = kwargs.get("right_dataset", "default_right")
        self.all_files = {"left": [], "right": []}
        cache_dir = Path(config.DEFAULT_HF_DATASETS_CACHE) / f"{left_dataset.replace('/', '_')}_{right_dataset.replace('/', '_')}"
        super().__init__(*args, **kwargs, cache_dir=cache_dir)

        self.left_name = self.config.left_dataset.split("/")[-1]
        self.right_name = self.config.right_dataset.split("/")[-1]
        self._relevant_partitions: Optional[List[Tuple[int, int]]] = None

    def _download_and_prepare(self, dl_manager, verification_mode, **prepare_split_kwargs):
        import pdb; pdb.set_trace()
        index_tables = self._download_and_load_index(dl_manager)
        import pdb; pdb.set_trace()
        matched_catalog = self.crossmatch_index_tables(*index_tables)
        import pdb; pdb.set_trace()
        self.download_matched_catalog(matched_catalog)

    def download_matched_catalog(self, matched_catalog: Table):
        left_object_ids = matched_catalog[f'{self.left_name}_object_id']
        right_object_ids = matched_catalog[f'{self.right_name}_object_id']
        files_to_download = {k: [f for f in v if f.endswith(".parquet") and not f.endswith("_index/index.parquet")] for k, v in self.all_files.items()}
        left_files = files_to_download[self.left_name]
        right_files = files_to_download[self.right_name]
        import pdb; pdb.set_trace()
        left_table = self._download_files(left_files, left_object_ids)
        right_table = self._download_files(right_files, right_object_ids)
        # todo sort tables by matched catalog


    def _download_files(self, files: list[str], object_ids) -> pa.Table:
        file_paths = [f"datasets/{f}" for f in files]
        hf_fs = HfFileSystem()

        tables = []

        for file_path in file_paths:
            table = pq.read_table(
                hf_fs.open(file_path, "rb"),
                filters=pc.field("object_id").isin(pa.array(object_ids))
            )
            tables.append(table)
        return pa.concat_tables(tables)

    def crossmatch_index_tables(self, left, right,
                                matching_radius : float = 1., 
                                ):
        left = Table.from_pandas(left.to_pandas())
        right = Table.from_pandas(right.to_pandas())

        left['sc'] = SkyCoord(left['ra'], 
                              left['dec'], unit='deg')
        
        right['sc'] = SkyCoord(right['ra'],
                               right['dec'], unit='deg')
        cat_left = left
        cat_right = right
        # Cross match the catalogs and restricting them to matches
        idx, sep2d, _ = cat_left['sc'].match_to_catalog_sky(cat_right['sc'])
        mask = sep2d < matching_radius*u.arcsec
        cat_left = cat_left[mask]
        cat_right = cat_right[idx[mask]]
        assert len(cat_left) == len(cat_right), "There was an error in the cross-matching."
        print("Initial number of matches: ", len(cat_left))
        matched_catalog = hstack([cat_left, cat_right], 
                                 table_names=[self.left_name, self.right_name],
                                 uniq_col_name='{table_name}_{col_name}')
        # Remove objects that were matched between the two catalogs but fall under different healpix indices
        mask = matched_catalog[f'{self.left_name}_healpix'] == matched_catalog[f'{self.right_name}_healpix']
        matched_catalog = matched_catalog[mask]
        print("Number of matches lost at healpix region borders: ", len(cat_left) - len(matched_catalog))
        print("Final size of cross-matched catalog: ", len(matched_catalog))

        # Adding default columns to respect format
        matched_catalog['object_id'] = matched_catalog[self.left_name+'_object_id']
        matched_catalog['ra'] = 0.5*(matched_catalog[self.left_name+'_ra'] +
                                     matched_catalog[self.right_name+'_ra'])
        matched_catalog['dec'] = 0.5*(matched_catalog[self.left_name+'_dec'] +
                                     matched_catalog[self.right_name+'_dec'])
        
        # Check that all matches have the same healpix index
        assert np.all(matched_catalog[self.left_name+'_healpix'] == matched_catalog[self.right_name+'_healpix']), "There was an error in the cross-matching."
        matched_catalog['healpix'] = matched_catalog[self.left_name+'_healpix']
        matched_catalog = matched_catalog.group_by(['healpix'])
        return matched_catalog

    def _download_and_load_index(self, dl_manager: DownloadManager) -> List[pa.Table]:
        """Download and load the _index partition into memory.

        Args:
            dl_manager: DownloadManager for handling downloads

        Returns:
            PyArrow Table containing the index data
        """
        index_urls = self._get_index_urls()

        if not index_urls:
            raise ValueError(f"No index files found in '{self.config.index_partition}' partition")

        # Download index files
        downloaded_index_files = dl_manager.download(index_urls)

        # Load all index files into a single table
        tables = []
        for file_path in downloaded_index_files:
            table = pq.read_table(file_path)
            tables.append(table)

        return tables

    def _get_index_urls(self) -> List[str]:
        """Get URLs/paths for all files in the _index partition.

        Uses HfFileSystem to list files in the repository and filter
        for those in the index partition.

        Returns:
            List of HF URLs (e.g., ["hf://datasets/org/dataset/train/_index/file.parquet"])
        """
        self.all_files = self._list_repository_files()

        # Filter for index partition files (under split_name/_index/)
        import pdb; pdb.set_trace()
        all_files_flat = self.all_files[self.left_name] + self.all_files[self.right_name]
        index_files = [
            f for f in all_files_flat
            if f.endswith("_index/index.parquet")
        ]
        index_urls = [f"hf://datasets/{f}" for f in index_files]

        return index_urls

    def _list_repo_files_single_ds(self, dataset_name):
        if (Path(config.DEFAULT_HF_DATASETS_CACHE) / dataset_name.replace("/", "_")).exists():
            base_path = Path(config.DEFAULT_HF_DATASETS_CACHE) / dataset_name.replace("/", "_")
            files = []
            for file_path in base_path.rglob("*"):
                if file_path.is_file():
                    files.append(str(file_path.relative_to(base_path)))
            return files
        else:
            repo_id = self._extract_repo_id_from_url(dataset_name)
            try:
                fs = HfFileSystem()
                files = fs.ls(f"datasets/{repo_id}", detail=False, recursive=True)
                # Strip the repo prefix
                files = [f.replace(f"datasets/", "") for f in files]
                return files
            except Exception as e:
                raise ValueError(f"Failed to list files in repository {repo_id}: {e}")


    def _list_repository_files(self) -> Dict[str, str]:
        """List all files in the dataset repository.

        Returns:
            List of file paths in the repository
        """
        files_left = self._list_repo_files_single_ds(self.config.left_dataset)
        files_right = self._list_repo_files_single_ds(self.config.right_dataset)
        return {self.left_name: files_left,
                self.right_name: files_right}

    def _extract_repo_id_from_url(self, url: str) -> str:
        """Extract repo_id from HuggingFace URL.

        Args:
            url: HF URL like "hf://datasets/org/dataset" or "https://huggingface.co/datasets/org/dataset"

        Returns:
            Repo ID like "org/dataset"
        """
        if url.startswith("hf://datasets/"):
            return url.replace("hf://datasets/", "").split("@")[0].rstrip("/")
        elif "huggingface.co/datasets/" in url:
            parts = url.split("huggingface.co/datasets/")[1]
            return parts.split("/")[0] + "/" + parts.split("/")[1]
        else:
            # Assume it's already a repo_id
            return url.rstrip("/")

    def _info(*args, **kwargs) -> DatasetInfo:
        """Return the dataset metadata and schema."""
        return DatasetInfo()
