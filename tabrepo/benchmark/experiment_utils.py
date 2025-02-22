from __future__ import annotations

from typing import Literal, Type

import pandas as pd
from autogluon_benchmark.tasks.task_wrapper import OpenMLTaskWrapper

from tabrepo.repository.repo_utils import convert_time_infer_s_from_batch_to_sample as _convert_time_infer_s_from_batch_to_sample
from tabrepo.utils.cache import AbstractCacheFunction, CacheFunctionPickle, CacheFunctionDummy
from tabrepo import EvaluationRepository
from tabrepo.benchmark.experiment_constructor import Experiment


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

    def run(
        self,
        datasets: list[str],
        folds: list[int],
        methods: list[Experiment],
        ignore_cache: bool,
    ) -> list:
        unknown_datasets = []
        for dataset in datasets:
            if dataset not in self._dataset_to_tid_dict:
                unknown_datasets.append(dataset)
        if unknown_datasets:
            raise ValueError(
                f"Dataset must be present in task_metadata!"
                f"\n\tInvalid Datasets: {unknown_datasets}"
                f"\n\t  Valid Datasets: {list(self._dataset_to_tid_dict.keys())}"
            )
        if len(datasets) != len(set(datasets)):
            raise AssertionError(f"Duplicate datasets present! Ensure all datasets are unique.")
        if len(folds) != len(set(folds)):
            raise AssertionError(f"Duplicate folds present! Ensure all folds are unique.")

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
        )

    def generate_repo_from_experiments(
        self,
        datasets: list[str],
        folds: list[int],
        methods: list[Experiment],
        ignore_cache: bool,
        convert_time_infer_s_from_batch_to_sample: bool = True,
    ) -> EvaluationRepository:
        """

        Parameters
        ----------
        datasets
        folds
        methods
        task_metadata
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
        )

        repo = self.repo_from_results(
            results_lst=results_lst,
            convert_time_infer_s_from_batch_to_sample=convert_time_infer_s_from_batch_to_sample,
        )

        return repo

    def repo_from_results(
        self,
        results_lst: list,
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

    Returns
    -------
    result_lst: list[dict], containing all metrics from fit() and predict() of all the given OpenML tasks
    """
    if cache_cls is None:
        cache_cls = CacheFunctionDummy
    if cache_cls_kwargs is None:
        cache_cls_kwargs = {}

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

                cacher = cache_cls(cache_name=cache_name, cache_path=expname, **cache_cls_kwargs)

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
