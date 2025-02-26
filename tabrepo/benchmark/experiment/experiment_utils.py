from __future__ import annotations

from typing import Any, Literal, Type

import pandas as pd
from autogluon_benchmark.tasks.task_wrapper import OpenMLTaskWrapper

from tabrepo.repository.repo_utils import convert_time_infer_s_from_batch_to_sample as _convert_time_infer_s_from_batch_to_sample
from tabrepo.utils.cache import AbstractCacheFunction, CacheFunctionPickle, CacheFunctionDummy
from tabrepo import EvaluationRepository
from tabrepo.benchmark.experiment.experiment_constructor import Experiment


# TODO: Inspect artifact folder to load all results without needing to specify them explicitly
#  generate_repo_from_dir(expname)
class ExperimentBatchRunner:
    def __init__(
        self,
        expname: str,
        task_metadata: pd.DataFrame,
        cache_cls: Type[AbstractCacheFunction] | None = CacheFunctionPickle,
        cache_cls_kwargs: dict | None = None,
        cache_path_format: Literal["name_first", "task_first"] = "name_first",
    ):
        """

        Parameters
        ----------
        expname
        cache_cls
        cache_cls_kwargs
        cache_path_format: {"name_first", "task_first"}, default "name_first"
            Determines the folder structure for artifacts.
            "name_first" -> {expname}/data/{method}/{tid}/{fold}/
            "task_first" -> {expname}/data/tasks/{tid}/{fold}/{method}/
        """
        cache_cls = CacheFunctionDummy if cache_cls is None else cache_cls
        cache_cls_kwargs = {} if cache_cls_kwargs is None else cache_cls_kwargs

        self.expname = expname
        self.task_metadata = task_metadata
        self.cache_cls = cache_cls
        self.cache_cls_kwargs = cache_cls_kwargs
        self.cache_path_format = cache_path_format
        self._dataset_to_tid_dict = task_metadata[['tid', 'dataset']].drop_duplicates(['tid', 'dataset']).set_index('dataset')['tid'].to_dict()

    @property
    def datasets(self) -> list[str]:
        return list(self._dataset_to_tid_dict.keys())

    def run(
        self,
        methods: list[Experiment],
        datasets: list[str],
        folds: list[int],
        ignore_cache: bool = False,
        mode: str = "local",
        s3_bucket: str = "prateek-ag",
    ) -> list[dict[str, Any]]:
        """

        Parameters
        ----------
        methods
        datasets
        folds
        ignore_cache: bool, default False
            If True, will run the experiments regardless if the cache exists already, and will overwrite the cache file upon completion.
            If False, will load the cache result if it exists for a given experiment, rather than running the experiment again.

        Returns
        -------
        results_lst: list[dict[str, Any]]
            A list of experiment run metadata dictionaries.
            Can pass into `exp_bach_runner.repo_from_results(results_lst=results_lst)` to generate an EvaluationRepository.

        """
        self._validate_datasets(datasets=datasets)
        self._validate_folds(folds=folds)

        tids = [self._dataset_to_tid_dict[dataset] for dataset in datasets]
        return run_experiments(
            expname=self.expname,
            tids=tids,
            folds=folds,
            methods=methods,
            task_metadata=self.task_metadata,
            ignore_cache=ignore_cache,
            cache_cls=self.cache_cls,
            cache_cls_kwargs=self.cache_cls_kwargs,
            cache_path_format=self.cache_path_format,
            mode=mode,
            s3_bucket=s3_bucket,
        )

    def load_results(
        self,
        methods: list[Experiment | str],
        datasets: list[str],
        folds: list[int],
        require_all: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Load results from the cache.

        Parameters
        ----------
        methods
        datasets
        folds
        require_all: bool, default True
            If True, will raise an exception if not all methods x datasets x folds have a cached result to load.
            If False, will return only the list of results with a cached result. This can be an empty list if no cached results exist.

        Returns
        -------
        results_lst
            The same output format returned by `self.run`

        """
        results_lst = []
        results_lst_exists = []
        results_lst_missing = []
        for method in methods:
            if isinstance(method, Experiment):
                method_name = method.name
            else:
                method_name = method
            for dataset in datasets:
                for fold in folds:
                    cache_exists = self._cache_exists(method_name=method_name, dataset=dataset, fold=fold)
                    cache_args = (method_name, dataset, fold)
                    if cache_exists:
                        results_lst_exists.append(cache_args)
                        print(method.name, dataset, fold)
                        print(f"\t{cache_exists}")
                    else:
                        results_lst_missing.append(cache_args)
        if require_all and results_lst_missing:
            raise AssertionError(
                f"Missing cached results for {len(results_lst_missing)}/{len(results_lst_exists) + len(results_lst_missing)} experiments! "
                f"\nTo load only the {len(results_lst_exists)} existing experiments, set `require_all=False`, "
                f"or call `exp_batch_runner.run(methods=methods, datasets=datasets, folds=folds)` to run the missing experiments."
                f"\nMissing experiments:\n\t{results_lst_missing}"
            )
        for method_name, dataset, fold in results_lst_exists:
            results_lst.append(self._load_result(method_name=method_name, dataset=dataset, fold=fold))
        return results_lst

    def generate_repo_from_experiments(
        self,
        datasets: list[str],
        folds: list[int],
        methods: list[Experiment],
        ignore_cache: bool,
        convert_time_infer_s_from_batch_to_sample: bool = True,
        mode="local",
        s3_bucket: str = "prateek-ag",
    ) -> EvaluationRepository:
        """

        Parameters
        ----------
        datasets
        folds
        methods
        ignore_cache
        convert_time_infer_s_from_batch_to_sample

        Returns
        -------
        EvaluationRepository

        """
        results_lst = self.run(
            datasets=datasets,
            folds=folds,
            methods=methods,
            ignore_cache=ignore_cache,
            mode=mode,
            s3_bucket=s3_bucket,
        )

        repo = self.repo_from_results(
            results_lst=results_lst,
            convert_time_infer_s_from_batch_to_sample=convert_time_infer_s_from_batch_to_sample,
        )

        return repo

    def repo_from_results(
        self,
        results_lst: list[dict[str, Any]],
        convert_time_infer_s_from_batch_to_sample: bool = True,  # FIXME: Remove this, it should be False eventually
    ) -> EvaluationRepository:
        configs_hyperparameters = self.get_configs_hyperparameters(results_lst=results_lst)

        results_baselines = [result["df_results"] for result in results_lst if result["simulation_artifacts"] is None]
        df_baselines = pd.concat(results_baselines, ignore_index=True) if results_baselines else None

        results_configs = [result for result in results_lst if result["simulation_artifacts"] is not None]

        results_lst_simulation_artifacts = [result["simulation_artifacts"] for result in results_configs]
        results_lst_df = [result["df_results"] for result in results_configs]

        df_configs = pd.concat(results_lst_df, ignore_index=True)
        if convert_time_infer_s_from_batch_to_sample:
            df_configs = _convert_time_infer_s_from_batch_to_sample(df=df_configs, task_metadata=self.task_metadata)

        # TODO: per-fold pred_proba_test and pred_proba_val (indices?)
        repo: EvaluationRepository = EvaluationRepository.from_raw(
            df_configs=df_configs,
            df_baselines=df_baselines,
            results_lst_simulation_artifacts=results_lst_simulation_artifacts,
            task_metadata=self.task_metadata,
            configs_hyperparameters=configs_hyperparameters,
        )

        return repo

    def get_configs_hyperparameters(self, results_lst: list[dict]) -> dict | None:
        configs_hyperparameters = {}
        for result in results_lst:
            if "method_metadata" in result and "model_hyperparameters" in result["method_metadata"]:
                method_name = result["framework"]
                if method_name in configs_hyperparameters:
                    continue
                method_metadata = result["method_metadata"]
                model_hyperparameters = method_metadata["model_hyperparameters"]
                model_cls = method_metadata.get("model_cls", None)
                model_type = method_metadata.get("model_type", None)
                name_prefix = method_metadata.get("name_prefix", None)

                configs_hyperparameters[method_name] = dict(
                    model_cls=model_cls,
                    model_type=model_type,
                    name_prefix=name_prefix,
                    hyperparameters=model_hyperparameters,
                )
        if not configs_hyperparameters:
            configs_hyperparameters = None
        return configs_hyperparameters

    def _cache_name(self, method_name: str, dataset: str, fold: int) -> str:
        # TODO: Windows? Use Path?
        tid = self._dataset_to_tid_dict[dataset]
        if self.cache_path_format == "name_first":
            cache_name = f"data/{method_name}/{tid}/{fold}/results"
        elif self.cache_path_format == "task_first":
            # Legacy format from early prototyping
            cache_name = f"data/tasks/{tid}/{fold}/{method_name}/results"
        else:
            raise ValueError(f"Unknown cache_path_format: {self.cache_path_format}")
        return cache_name

    def _cache_exists(self, method_name: str, dataset: str, fold: int) -> bool:
        cacher = self._get_cacher(method_name=method_name, dataset=dataset, fold=fold)
        return cacher.exists

    def _load_result(self, method_name: str, dataset: str, fold: int) -> dict[str, Any]:
        cacher = self._get_cacher(method_name=method_name, dataset=dataset, fold=fold)
        return cacher.load_cache()

    def _get_cacher(self, method_name: str, dataset: str, fold: int) -> AbstractCacheFunction:
        cache_name = self._cache_name(method_name=method_name, dataset=dataset, fold=fold)
        cacher = self.cache_cls(cache_name=cache_name, cache_path=self.expname, **self.cache_cls_kwargs)
        return cacher

    def _validate_datasets(self, datasets: list[str]):
        unknown_datasets = []
        for dataset in datasets:
            if dataset not in self._dataset_to_tid_dict:
                unknown_datasets.append(dataset)
        if unknown_datasets:
            raise ValueError(
                f"Dataset must be present in task_metadata!"
                f"\n\tInvalid Datasets: {unknown_datasets}"
                f"\n\t  Valid Datasets: {self.datasets}"
            )
        if len(datasets) != len(set(datasets)):
            raise AssertionError(f"Duplicate datasets present! Ensure all datasets are unique.")

    def _validate_folds(self, folds: list[int]):
        if len(folds) != len(set(folds)):
            raise AssertionError(f"Duplicate folds present! Ensure all folds are unique.")


def run_experiments(
    expname: str,
    tids: list[int],
    folds: list[int],
    methods: list[Experiment],
    task_metadata: pd.DataFrame,
    ignore_cache: bool,
    cache_cls: Type[AbstractCacheFunction] | None = CacheFunctionPickle,
    cache_cls_kwargs: dict = None,
    cache_path_format: Literal["name_first", "task_first"] = "name_first",
    mode: str = "local",
    s3_bucket: str = "prateek-ag",
) -> list[dict]:
    """

    Parameters
    ----------
    expname: str, Name of the experiment given by the user
    tids: list[int], List of OpenML task IDs given by the user
    folds: list[int], Number of folds present for the given task
    methods: list[Experiment], Models used for fit() and predict() in this experiment
    task_metadata: pd.DataFrame, OpenML task metadata
    ignore_cache: bool, whether to use cached results (if present)
    cache_cls: WIP
    cache_cls_kwargs: WIP
    cache_path_format: {"name_first", "task_first"}, default "name_first"
    mode: {"local", "aws"}, default "local"
    S3_bucket: str, default "prateek-ag" works only for aws mode, stores artifacts in the given bucket

    Returns
    -------
    result_lst: list[dict], containing all metrics from fit() and predict() of all the given OpenML tasks
    """
    if cache_cls is None:
        cache_cls = CacheFunctionDummy
    if cache_cls_kwargs is None:
        cache_cls_kwargs = {}

    # Modify cache path based on mode
    if mode == "local":
        base_cache_path = expname
    else:
        base_cache_path = f"s3://{s3_bucket}/{expname}"

    methods_og = methods
    methods = []
    for method in methods_og:
        # TODO: remove tuple input option, doing it to keep old scripts working
        if not isinstance(method, Experiment):
            method = Experiment(name=method[0], method_cls=method[1], method_kwargs=method[2])
        methods.append(method)

    unique_names = set()
    for method in methods:
        if method.name in unique_names:
            raise AssertionError(f"Duplicate experiment name found. All names must be unique. name: {method.name}")
        unique_names.add(method.name)

    # FIXME: dataset or name? Where does `dataset` come from, why can it be different from `name`?
    #  Using dataset for now because for some datasets like "GAMETES", the name is slightly different with `.` in `name` being replaced with `_` in `dataset`.
    #  This is probably because `.` isn't a valid name in a file in s3.
    #  TODO: What if `dataset` doesn't exist as a column? Maybe fallback to `name`? Or do the `name` -> `dataset` conversion, or use tid.
    dataset_name_column = "dataset"
    dataset_names = [task_metadata[task_metadata["tid"] == tid][dataset_name_column].iloc[0] for tid in tids]
    print(
        f"Running Experiments for expname: '{expname}'..."
        f"\n\tFitting {len(tids)} datasets and {len(folds)} folds for a total of {len(tids) * len(folds)} tasks"
        f"\n\tFitting {len(methods)} methods on {len(tids) * len(folds)} tasks for a total of {len(tids) * len(folds) * len(methods)} jobs..."
        f"\n\tTIDs    : {tids}"
        f"\n\tDatasets: {dataset_names}"
        f"\n\tFolds   : {folds}"
        f"\n\tMethods : {[method.name for method in methods]}"
    )
    result_lst = []
    num_datasets = len(tids)
    for i, tid in enumerate(tids):
        task = None  # lazy task loading
        task_name = task_metadata[task_metadata["tid"] == tid][dataset_name_column].iloc[0]
        print(f"Starting Dataset {i+1}/{num_datasets}...")
        for fold in folds:
            for method in methods:
                if cache_path_format == "name_first":
                    cache_name = f"data/{method.name}/{tid}/{fold}/results"
                elif cache_path_format == "task_first":
                    # Legacy format from early prototyping
                    cache_name = f"data/tasks/{tid}/{fold}/{method.name}/results"
                else:
                    raise ValueError(f"Invalid cache_path_format: {cache_path_format}")
                print(
                    f"\tFitting {task_name} on fold {fold} for method {method.name}"
                )

                cacher = cache_cls(cache_name=cache_name, cache_path=base_cache_path, **cache_cls_kwargs)

                if task is None:
                    if ignore_cache or not cacher.exists:
                        task = OpenMLTaskWrapper.from_task_id(task_id=tid)

                out = method.run(
                    task=task,
                    fold=fold,
                    task_name=task_name,
                    cacher=cacher,
                    ignore_cache=ignore_cache,
                )
                result_lst.append(out)

    return result_lst


def convert_leaderboard_to_configs(leaderboard: pd.DataFrame, minimal: bool = True) -> pd.DataFrame:
    df_configs = leaderboard.rename(columns=dict(
        time_fit="time_train_s",
        time_predict="time_infer_s",
        test_error="metric_error",
        eval_metric="metric",
        val_error="metric_error_val",
    ))
    if minimal:
        minimal_columns = [
            "dataset",
            "fold",
            "framework",
            "metric_error",
            "metric",
            "problem_type",
            "time_train_s",
            "time_infer_s",
            "tid",
        ]
        if "metric_error_val" in df_configs:
            minimal_columns.append("metric_error_val")
        df_configs = df_configs[minimal_columns]
    return df_configs
