from pathlib import Path
import os
from typing import Dict, List, Tuple

from ._configs import load_configs
from ._results import load_results
from ._download import download_zs_metadata


class Paths:
    project_root: Path = Path(__file__).parent.parent.parent
    data_root: Path = project_root / 'data'
    data_root_cache: Path = Path.home() / ".cache" / "tabrepo" / "data"
    results_root_cache: Path = data_root_cache / "results"

    @staticmethod
    def abs_to_rel(path: str, relative_to: Path = project_root) -> str:
        relative_to_split = str(relative_to)
        if relative_to_split[-1] != os.sep:
            relative_to_split += os.sep
        assert path.startswith(relative_to_split)
        path_relative = path.split(relative_to_split, 1)[1]
        return path_relative

    @staticmethod
    def rel_to_abs(path: str, relative_to: Path = project_root) -> str:
        return str(relative_to / Path(path))

    @staticmethod
    def s3_to_local_tuple_list_to_dict(
            s3_to_local_tuple_list: List[Tuple[str, str]],
            relative_to: Path = data_root,
    ) -> Dict[str, str]:
        return {Paths.abs_to_rel(val, relative_to=relative_to): key for key, val in s3_to_local_tuple_list}
