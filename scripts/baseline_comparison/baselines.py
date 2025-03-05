import copy
import itertools
from typing import List

import numpy as np
from dataclasses import dataclass

from tqdm import tqdm

from tabrepo.portfolio.zeroshot_selection import zeroshot_configs
from tabrepo.repository import EvaluationRepository
from tabrepo.utils.parallel_for import parallel_for

default_ensemble_size = 40
n_portfolios_default = 200
default_runtime = 3600 * 4

backup_fast_config = "ExtraTrees_c1_BAG_L1"


@dataclass
class ResultRow:
    dataset: str
    fold: int
    method: str
    test_error: float
    rank: float
    normalized_error: float
    time_train_s: float
    time_infer_s: float
    metric_error_val: float = None
    config_selected: list = None
    seed: int = None
    metadata: dict = None


def evaluate_configs(
        repo: EvaluationRepository,
        configs: List[str],
        rank_scorer,
        normalized_scorer,
        tid: int,
        folds: List[int],
        method: str,
        ensemble_size: int = default_ensemble_size,
        ensemble_kwargs=None,
        time_limit: float | None = None,
        fit_order="original",
        seed: int = 0,
) -> List[ResultRow]:
    """

    :param repo:
    :param configs:
    :param rank_scorer:
    :param normalized_scorer:
    :param tid:
    :param method:
    :param ensemble_size:
    :param folds:
    :return: list of results for each fold in `folds` evaluated on task `tid` with `config_selected` configurations
    """
    if ensemble_size is None:
        ensemble_size = default_ensemble_size

    dataset = repo.tid_to_dataset(tid=tid)

    rows = []
    for fold in folds:
        df_metrics, ensemble_weights = repo.evaluate_ensemble(
            dataset=dataset,
            fold=fold,
            configs=configs,
            fit_order=fit_order,
            seed=seed,
            ensemble_size=ensemble_size,
            ensemble_kwargs=ensemble_kwargs,
            time_limit=time_limit,
            rank=False,
        )
        assert len(df_metrics) == 1
        metrics = df_metrics.iloc[0]
        configs_selected = [c for c in configs if c in ensemble_weights.columns]

        task = repo.task_name(dataset=dataset, fold=fold)

        metric_error = metrics["metric_error"]
        metric_error_val = metrics["metric_error_val"]
        time_train_s = metrics["time_train_s"]
        time_infer_s = metrics["time_infer_s"]

        rows.append(ResultRow(
            dataset=dataset,
            fold=fold,
            method=method,
            test_error=metric_error,
            rank=rank_scorer.rank(task, metric_error),
            normalized_error=normalized_scorer.rank(task, metric_error),
            time_train_s=time_train_s,
            time_infer_s=time_infer_s,
            metric_error_val=metric_error_val,
            config_selected=configs_selected,
            seed=seed,
            metadata=dict(
                n_iterations=ensemble_size,
                time_limit=time_limit,
            )
        ))
    return rows


def framework_name(framework_type, max_runtime=None, ensemble_size=default_ensemble_size, tuned: bool=True, all: bool = False, prefix: str = None) -> str:
    method = framework_type if framework_type else "All"
    if prefix is None:
        prefix = ""
    if all:
        method = "All"
    if not tuned:
        suffix = " (default)"
    else:
        suffix = " (tuned + ensemble)" if ensemble_size > 1 else " (tuned)"
        suffix += time_suffix(max_runtime=max_runtime)
    method = f"{method}{prefix}{suffix}"
    return method


def framework_default_results(repo: EvaluationRepository,
                              dataset_names: List[str],
                              framework_types: List[str],
                              n_eval_folds: int,
                              rank_scorer,
                              normalized_scorer,
                              engine: str,
                              **kwargs) -> List[ResultRow]:
    """
    :return: evaluations of default models (e.g. 'CatBoost_c1_BAG_L1') and the best/ensemble of all default models
    """

    def evaluate_tid(dataset_name, default, repo, rank_scorer, normalized_scorer):
        name, configs, ensemble_size = default
        return evaluate_configs(
            repo=repo,
            rank_scorer=rank_scorer,
            normalized_scorer=normalized_scorer,
            configs=configs,
            ensemble_size=ensemble_size,
            tid=repo.dataset_to_tid(dataset_name),
            folds=range(n_eval_folds),
            method=name,
        )

    defaults = [
        (framework_name(framework_type, tuned=False), [f'{framework_type}_c1_BAG_L1'], 1)
        for framework_type in framework_types
    ]

    list_rows = parallel_for(
        evaluate_tid,
        inputs=list(itertools.product(dataset_names, defaults)),
        context=dict(repo=repo, rank_scorer=rank_scorer, normalized_scorer=normalized_scorer),
        engine=engine,
    )
    return [x for l in list_rows for x in l]


def framework_best_results(
        repo: EvaluationRepository,
        dataset_names: List[str],
        framework_types: List[str],
        n_eval_folds: int,
        rank_scorer,
        normalized_scorer,
        all: bool = False,
        max_runtimes: float = [3600],
        ensemble_size: int = default_ensemble_size,
        method_prefix: str = None,
        engine: str = 'ray',
        random_state: int = 0,
        **kwargs) -> List[ResultRow]:
    """
    Evaluates best configurations among `n_configs` random draws and ensemble built with `ensemble_size`
    configurations with highest validation scores among the `n_configs` configurations.
    """

    def evaluate_tid(dataset_name, max_runtime, framework_type, ensemble_size, repo, rank_scorer, normalized_scorer, random_state, all):
        tid = repo.dataset_to_tid(dataset_name)
        rows = []

        for fold in range(n_eval_folds):
            df_score_val = repo._zeroshot_context.df_configs_ranked

            # gets rows with desired task and framework
            mask = (df_score_val['dataset'] == dataset_name) & (df_score_val.fold == fold)
            if framework_type:
                if isinstance(framework_type, list):
                    mask &= (df_score_val.framework.str.contains('|'.join(framework_type)))
                else:
                    mask &= (df_score_val.framework.str.contains(framework_type))
            df_sub = df_score_val[mask]
            configs = df_sub["framework"].tolist()

            # evaluate them
            rows += evaluate_configs(
                repo=repo,
                rank_scorer=rank_scorer,
                normalized_scorer=normalized_scorer,
                configs=configs,
                ensemble_size=ensemble_size,
                time_limit=max_runtime,
                fit_order="random",
                seed=random_state,
                tid=tid,
                folds=[fold],
                method=framework_name(framework_type, max_runtime, ensemble_size, tuned=True, all=all, prefix=method_prefix),
            )
        return rows

    ensemble_sizes = [1, ensemble_size]
    list_rows = parallel_for(
        evaluate_tid,
        inputs=list(itertools.product(dataset_names, max_runtimes, framework_types, ensemble_sizes)),
        context=dict(repo=repo, rank_scorer=rank_scorer, normalized_scorer=normalized_scorer, random_state=random_state, all=all),
        engine=engine,
    )
    return [x for l in list_rows for x in l]


def automl_results(repo: EvaluationRepository, dataset_names: List[str], n_eval_folds: int, rank_scorer,
                   normalized_scorer, **kwargs) -> List[ResultRow]:
    """
    :return: evaluation of AutoGluon medium/high/best quality.
    """
    automl_df = copy.deepcopy(repo._zeroshot_context.df_baselines)

    rows_automl = []
    for dataset in tqdm(dataset_names):
        tid = repo.dataset_to_tid(dataset)
        for fold in range(n_eval_folds):
            task = repo.task_name(dataset=dataset, fold=fold)
            automl_df_fold = automl_df[automl_df['task'] == task]
            task_automl_dict = automl_df_fold.T.to_dict()

            for k, v in task_automl_dict.items():
                assert tid == v['tid']
                metric_error = v['metric_error']
                rows_automl.append(ResultRow(
                    dataset=dataset,
                    fold=v['fold'],
                    method=v['framework'],
                    test_error=metric_error,
                    rank=rank_scorer.rank(task, metric_error),
                    normalized_error=normalized_scorer.rank(task, metric_error),
                    time_train_s=v['time_train_s'],
                    time_infer_s=v['time_infer_s'],
                ))

    return rows_automl


def time_suffix(max_runtime: float) -> str:
    if max_runtime:
        if max_runtime >= 3600:
            str_num_hours = f"{int(max_runtime / 3600)}" if max_runtime % 3600 == 0 else f"{max_runtime / 3600:0.2f}"
            return f" ({str_num_hours}h)"
        else:
            str_num_mins = f"{int(max_runtime / 60)}" if max_runtime % 60 == 0 else f"{max_runtime / 60:0.2f}"
            return f" ({str_num_mins}m)"
    else:
        return ""


def zeroshot_name(
        n_portfolio: int = n_portfolios_default, n_ensemble: int = None, n_training_dataset: int = None,
        n_training_fold: int = None, n_training_config: int = None,
        max_runtime: float = default_runtime, prefix: str = None, n_ensemble_in_name: bool = False,
        max_models: int = None, max_models_per_type: int = None,
):
    """
    :return: name of the zeroshot method such as Zeroshot-N20-C40 if n_training_dataset or n_training_folds are not
    None, suffixes "-D{n_training_dataset}" and "-S{n_training_folds}" are added, for instance "Zeroshot-N20-C40-D30-S5"
    would be the name if n_training_dataset=30 and n_training_fold=5
    """
    if max_models_per_type is not None and isinstance(max_models_per_type, str) and max_models_per_type == "auto":
        max_models_per_type = 0  # FIXME: HACK, using 0 because otherwise regex parsing fails if "auto". Switch to not rely on name parsing in experiments.
    if prefix is None:
        prefix = ""
    suffix = [
        f"-{letter}{x}" if x is not None else ""
        for letter, x in
        [("N", n_portfolio), ("D", n_training_dataset), ("S", n_training_fold), ("M", n_training_config), ("X", max_models), ("Z", max_models_per_type)]
    ]
    # if n_ensemble:
    #     suffix += f"-C{n_ensemble}"
    suffix = "".join(suffix)
    if n_ensemble_in_name and n_ensemble is not None:
        suffix += f"-E{n_ensemble}"
    if n_ensemble is None or n_ensemble > 1:
        suffix += " (ensemble)"
    suffix += time_suffix(max_runtime)
    return f"Portfolio{prefix}{suffix}"


def filter_configurations_above_budget(repo, test_tid, configs, max_runtime, quantile: float = 0.95):
    # Filter configurations which respects the constrain less than `quantile` fraction of the time
    assert 0<= quantile <= 1
    dd = repo._zeroshot_context.df_configs_ranked
    dd = dd[dd.tid != test_tid]
    df_configs_runtime = dd.pivot_table(
        index="framework", columns="tid", values="time_train_s"
    ).quantile(q=quantile, axis=1).sort_values()

    n_initial_configs = len(configs)
    configs_fast_enough = set(df_configs_runtime[df_configs_runtime < max_runtime].index.tolist())
    configs = [c for c in configs if c in configs_fast_enough]
    # print(f"kept only {len(configs)} from initial {n_initial_configs} for runtime {max_runtime}")
    return configs


def zeroshot_results(
        repo: EvaluationRepository,
        dataset_names: List[str],
        rank_scorer,
        normalized_scorer,
        n_eval_folds: int,
        framework_types: List[str] | None = None,
        configs: list[str] | None = None,
        n_ensembles: List[int] = [None],
        n_portfolios: List[int] = [n_portfolios_default],
        n_training_datasets: List[int] = [None],
        n_training_folds: List[int] = [None],
        n_training_configs: List[int] = [None],
        n_max_models: List[int] = [None],
        n_max_models_per_type: List[int] = [None],
        max_runtimes: List[float] = [default_runtime],
        engine: str = "ray",
        seeds: list = [0],
        method_prefix: str = None,
        n_ensemble_in_name: bool = False,
) -> list[ResultRow]:
    """
    :param dataset_names: list of dataset to use when fitting zeroshot
    :param n_eval_folds: number of folds to consider for evaluation
    :param n_ensembles: number of caruana sizes to consider
    :param n_portfolios: number of folds to use when fitting zeroshot
    :param n_training_datasets: number of dataset to use when fitting zeroshot
    :param n_training_folds: number of folds to use when fitting zeroshot
    :param n_training_configs: number of configurations available when fitting zeroshot TODO per framework
    :param n_max_models: maximum number of models considered when fitting ensemble
    :param n_max_models_per_type: maximum number of models for each framework considered when fitting ensemble
    :param max_runtimes: max runtime available when evaluating zeroshot configuration at test time
    :param engine: engine to use, must be "sequential", "joblib" or "ray"
    :param seeds: the seeds for the random number generator used for shuffling the configs
    :return: evaluation obtained on all combinations
    """

    def evaluate_dataset(test_dataset, n_portfolio, n_ensemble, n_training_dataset, n_training_fold, n_training_config,
                         max_runtime, max_models, max_models_per_type, seed: int, repo: EvaluationRepository, df_rank, rank_scorer, normalized_scorer,
                         model_frameworks) -> list[ResultRow]:
        method_name = zeroshot_name(
            n_portfolio=n_portfolio,
            n_ensemble=n_ensemble,
            n_training_dataset=n_training_dataset,
            n_training_fold=n_training_fold,
            max_runtime=max_runtime,
            n_training_config=n_training_config,
            prefix=method_prefix,
            n_ensemble_in_name=n_ensemble_in_name,
            max_models=max_models,
            max_models_per_type=max_models_per_type
        )

        rng = np.random.default_rng(seed=seed)

        # restrict number of evaluation fold
        if n_training_fold is None:
            n_training_fold = n_eval_folds

        # gets all tids that are possible available
        test_tid = repo.dataset_to_tid(test_dataset)
        available_tids = [repo.dataset_to_tid(dataset) for dataset in dataset_names if dataset != test_dataset]
        rng.shuffle(available_tids)
        if n_training_dataset is None:
            n_training_dataset = len(available_tids)

        # restrict number of training tid availables to fit
        selected_tids = set(available_tids[:n_training_dataset])

        # restrict number of configurations available to fit
        configs = []
        for models_framework in model_frameworks.values():
            if not (n_training_config) or len(models_framework) <= n_training_config:
                configs += models_framework
            else:
                configs += list(rng.choice(models_framework, n_training_config, replace=False))

        # Randomly shuffle the config order with the passed seed
        configs = list(rng.choice(configs, len(configs), replace=False))

        # # exclude configurations from zeroshot selection whose runtime exceeds runtime budget by large amount
        if max_runtime:
            configs = filter_configurations_above_budget(repo, test_tid, configs, max_runtime)

        df_rank = df_rank.copy().loc[configs]

        # collects all tasks that are available
        train_tasks = []
        for task in df_rank.columns:
            tid, fold = task.split("_")
            if int(tid) in selected_tids and int(fold) < n_training_fold:
                train_tasks.append(task)

        # fit zeroshot portfolio on all available tasks
        indices = zeroshot_configs(-df_rank[train_tasks].values.T, n_portfolio)
        portfolio_configs = [df_rank.index[i] for i in indices]
        # TODO: Technically we should exclude data from the fold when computing the average runtime and also pass the
        #  current fold when filtering by runtime.
        # portfolio_configs = sort_by_runtime(repo=repo, config_names=portfolio_configs)

        if max_runtime is None:
            max_runtime = default_runtime

        ensemble_kwargs = {
            "max_models": max_models,
            "max_models_per_type": max_models_per_type,
        }

        results_row_lst: list[ResultRow] = evaluate_configs(
            repo=repo,
            rank_scorer=rank_scorer,
            normalized_scorer=normalized_scorer,
            configs=portfolio_configs,
            ensemble_size=n_ensemble,
            ensemble_kwargs=ensemble_kwargs,
            tid=test_tid,
            time_limit=max_runtime,
            method=method_name,
            folds=range(n_eval_folds),
            seed=seed,
        )

        for results_row in results_row_lst:
            results_row.metadata["n_portfolio"] = n_portfolio

        return results_row_lst

    dd = repo._zeroshot_context.df_configs_ranked
    # df_rank = dd.pivot_table(index="framework", columns="dataset", values="score_val").rank()
    # TODO use normalized scores
    df_rank = dd.pivot_table(index="framework", columns="task", values="metric_error").rank(ascending=False)
    df_rank.fillna(value=np.nanmax(df_rank.values), inplace=True)
    assert not any(df_rank.isna().values.reshape(-1))

    if configs is None:
        configs = repo.configs()
    if framework_types is None:
        model_frameworks = {
            "all": sorted(configs)
        }
    else:
        model_frameworks = {
            framework: sorted([x for x in configs if framework in x])
            for framework in framework_types
        }

    list_rows = parallel_for(
        evaluate_dataset,
        inputs=list(itertools.product(dataset_names, n_portfolios, n_ensembles, n_training_datasets, n_training_folds,
                                      n_training_configs, max_runtimes, n_max_models, n_max_models_per_type, seeds)),
        context=dict(repo=repo, df_rank=df_rank, rank_scorer=rank_scorer, normalized_scorer=normalized_scorer,
                     model_frameworks=model_frameworks),
        engine=engine,
    )
    return [x for l in list_rows for x in l]
