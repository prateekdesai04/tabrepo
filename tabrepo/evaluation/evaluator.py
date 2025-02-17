from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from autogluon.common.savers import save_pd

from ..repository import EvaluationRepository, EvaluationRepositoryCollection
from ..repository.repo_utils import convert_time_infer_s_from_sample_to_batch


# TODO: This class is WIP.
# TODO: Add unit tests
class Evaluator:
    """
    Computes metrics and statistics to compare methods.
    """
    def __init__(
        self,
        repo: EvaluationRepository | EvaluationRepositoryCollection,
    ):
        self.repo = repo

    # TODO: repo time_infer_s is per row, results_df is the total time for all rows, need to align later
    # TODO: Error if unknown configs/baselines requested
    # TODO: Add fillna
    # TODO: Docstring
    # Q:Whether to keep these functions a part of TabRepo or keep them separate as a part of new fit()-package
    def compare_metrics(
        self,
        results_df: pd.DataFrame = None,
        datasets: list[str] = None,
        folds: list[int] = None,
        configs: list[str] = None,
        baselines: list[str] = None,
        convert_from_sample_to_batch: bool = False,
    ) -> pd.DataFrame:
        if datasets is None:
            datasets = self.repo.datasets()
        columns = ["metric_error", "time_train_s", "time_infer_s", "metric", "problem_type"]

        if results_df is not None:
            df_exp = results_df.reset_index().set_index(["dataset", "fold", "framework"])[columns]
        else:
            df_exp = None

        # Dropping task column in df_tr
        df_tr = self.repo._zeroshot_context.df_configs.set_index(["dataset", "fold", "framework"])[columns]

        mask = df_tr.index.get_level_values("dataset").isin(datasets)
        if folds is not None:
            mask = mask & df_tr.index.get_level_values("fold").isin(folds)
        if configs is not None:
            mask = mask & df_tr.index.get_level_values("framework").isin(configs)
        df_tr = df_tr[mask]

        if self.repo.task_metadata is not None and convert_from_sample_to_batch:
            df_tr = convert_time_infer_s_from_sample_to_batch(df_tr, repo=self.repo)

        if self.repo._zeroshot_context.df_baselines is not None:
            df_baselines = self.repo._zeroshot_context.df_baselines.set_index(["dataset", "fold", "framework"])[columns]

            mask = df_baselines.index.get_level_values("dataset").isin(datasets)
            if folds is not None:
                mask = mask & df_baselines.index.get_level_values("fold").isin(folds)
            if baselines is not None:
                mask = mask & df_baselines.index.get_level_values("framework").isin(baselines)
            df_baselines = df_baselines[mask]

            if self.repo.task_metadata is not None and convert_from_sample_to_batch:
                df_baselines = convert_time_infer_s_from_sample_to_batch(df_baselines, repo=self.repo)
        else:
            if baselines:
                raise AssertionError(f"Baselines specified but no baseline methods exist! (baselines={baselines})")
            df_baselines = None

        df = pd.concat([df_exp, df_tr, df_baselines], axis=0)
        df = df.sort_index()

        return df

    # TODO: Prototype, find a better way to do this
    # TODO: Docstring
    def compute_avg_config_prediction_delta(
        self,
        configs: list[str],
        datasets: list[str] = None,
        folds: list[int] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """


        Parameters
        ----------
        configs
        datasets
        folds

        Returns
        -------
        delta_avg_abs_mean : pd.DataFrame
            The per-task normalized mean of the absolute delta in cells between two configs' test predictions.
            Maximum value is 1 (maximum disagreement), minimum value is 0 (identical configs).
        delta_avg_std : pd.DataFrame
            The per-task normalized mean of the standard deviation of the delta in cells between two configs' test predictions.
            Maximum value is 1 (maximum disagreement), minimum value is 0 (identical configs).
        """
        if datasets is None:
            datasets = self.repo.datasets()
        if folds is None:
            folds = self.repo.folds
        import numpy as np
        delta_comparison = {}
        delta_std_comparison = {}
        for dataset in datasets:
            for fold in folds:
                delta_comparison[(dataset, fold)] = {}
                delta_std_comparison[(dataset, fold)] = {}
                print(f"{dataset} | {fold}")
                # TODO: Only load y_pred_test 1 time for each task+config
                # TODO: don't need to do (x, y) and (y, x)
                for compare_conf1 in configs:
                    for compare_conf2 in configs:
                        if compare_conf1 == compare_conf2:
                            continue
                        print(f"{compare_conf1} vs {compare_conf2}")
                        y_pred_test1 = self.repo.predict_test(dataset=dataset, fold=fold, config=compare_conf1)
                        y_pred_test2 = self.repo.predict_test(dataset=dataset, fold=fold, config=compare_conf2)
                        delta = y_pred_test2 - y_pred_test1
                        # print(delta)
                        mean = np.mean(delta)
                        abs_mean = np.mean(np.abs(delta))
                        stddev = np.std(delta)
                        print(f"\t{abs_mean:.3f}\t{stddev:.3f}")
                        delta_comparison[(dataset, fold)][(compare_conf1, compare_conf2)] = abs_mean
                        delta_std_comparison[(dataset, fold)][(compare_conf1, compare_conf2)] = stddev
                # normalize
                max_abs_mean = 0
                max_std = 0
                for k, v in delta_comparison[(dataset, fold)].items():
                    max_abs_mean = max(max_abs_mean, v)
                for k, v in delta_std_comparison[(dataset, fold)].items():
                    max_std = max(max_std, v)
                for k in delta_comparison[(dataset, fold)].keys():
                    if max_abs_mean != 0:
                        delta_comparison[(dataset, fold)][k] /= max_abs_mean
                    if max_std != 0:
                        delta_std_comparison[(dataset, fold)][k] /= max_std
                # FIXME: FINISH

        delta_avg_abs_mean = pd.DataFrame(delta_comparison).mean(axis=1).unstack().fillna(0)
        delta_avg_std = pd.DataFrame(delta_std_comparison).mean(axis=1).unstack().fillna(0)

        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)

        pca.fit(delta_avg_abs_mean)
        delta_avg_abs_mean_projection = pca.transform(delta_avg_abs_mean)

        # FIXME: Make plotting optional
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        # config_names = list(delta_avg_abs_mean.index)
        delta_avg_abs_mean_projection = pd.DataFrame(delta_avg_abs_mean_projection, index=delta_avg_abs_mean.index)
        for config in delta_avg_abs_mean_projection.index:
            ax.scatter(delta_avg_abs_mean_projection.loc[config, 0], delta_avg_abs_mean_projection.loc[config, 1], label=config)
        ax.legend()
        ax.grid(True)
        plt.savefig("pca_projection_test")

        return delta_avg_abs_mean, delta_avg_std

    # TODO: Rename to something better?
    def plot_overall_rank_comparison(
        self,
        results_df: pd.DataFrame,
        save_dir: str,
        evaluator_kwargs: dict = None,
        calibration_framework: str = None,
    ) -> "EvaluatorOutput":
        """
        Requires `autogluon_benchmark` to be installed.

        Parameters
        ----------
        results_df: pd.DataFrame
            The input data to calculate metrics with.
            An easy way to obtain a valid `results_df` is to call `evaluator.compare_metrics(...)`
            It should have a multi-index of (dataset, fold, framework), with the following columns:
                metric_error: float
                metric: str
                time_train_s: float
                time_infer_s: float
                problem_type: str
        save_dir: str
            The local directory to save comparison results and figures to.
        evaluator_kwargs: dict, default = None
            The evaluator kwargs.
        calibration_framework: str, default = None
            The framework to fix at 1000 elo.

        Returns
        -------
        EvaluatorOutput object from autogluon_benchmark
        """
        try:
            from autogluon_benchmark.evaluation.evaluator import Evaluator as Eval
            from autogluon_benchmark.plotting.plotter import Plotter
        except ImportError:
            raise ImportError(f"To use `Evaluator.plot_overall_rank_comparison, you must first install autogluon_benchmark.")
        if evaluator_kwargs is None:
            evaluator_kwargs = {}
        results_df = results_df.reset_index().copy()
        results_df["tid"] = results_df["dataset"].apply(self.repo.dataset_to_tid)
        evaluator = Eval(task_metadata=self.repo.task_metadata, **evaluator_kwargs)
        evaluator_output = evaluator.transform(results_df)
        output_path = Path(save_dir)
        figure_savedir = str(output_path / "figures")
        save_pd.save(path=str(output_path / "results.csv"), df=results_df)
        save_pd.save(path=str(output_path / "results_ranked_agg.csv"), df=evaluator_output.results_ranked_agg)
        save_pd.save(path=str(output_path / "results_ranked.csv"), df=evaluator_output.results_ranked)

        plotter = Plotter(
            results_ranked_fillna_df=evaluator_output.results_ranked,
            results_ranked_df=evaluator_output.results_ranked,
            save_dir=figure_savedir,
            show=False,
        )

        plotter.plot_all(
            calibration_framework=calibration_framework,
            calibration_elo=1000,
            BOOTSTRAP_ROUNDS=100,  # Reduce this to lower values for a faster execution. Use 1000 for the final plot.
            plot_critical_difference=False,
        )

        return evaluator_output

    # TODO: aggregate_config_family: bool
    # TODO: sort rows by size? color by problem type?
    def plot_ensemble_weights(
        self,
        df_ensemble_weights: pd.DataFrame,
        aggregate_folds: bool = True,
        sort_by_mean: bool = True,
        include_mean: bool = True,
        **kwargs,
    ):
        """

        Parameters
        ----------
        df_ensemble_weights : pd.DataFrame
            The 2nd output object of `repo.evaluate_ensembles(...)
        aggregate_folds : bool, default True
            If True, averages folds of datasets together into single rows representing a dataset.
            If False, each fold of each dataset will be its own row.
        sort_by_mean : bool, default True
            If True, will sort columns by the mean value of the column.
            If False, columns will remain in the original order.
        include_mean : bool, default True
            If True, will add a row at the bottom with label "mean" representing the mean of the config weights across all tasks.
            NaN values are considered 0 for the purposes of calculating the mean.
        **kwargs
            Passed to the `create_heatmap` function

        Returns
        -------
        plt

        """
        df_ensemble_weights = copy.deepcopy(df_ensemble_weights)
        if aggregate_folds:
            df_ensemble_weights = df_ensemble_weights.groupby(level='dataset').mean()
        else:
            index_new = list(df_ensemble_weights.index.to_flat_index())
            index_new = [str(t[0]) + "_" + str(t[1]) for t in index_new]
            df_ensemble_weights.index = index_new

        if sort_by_mean:
            s = df_ensemble_weights.sum()
            df_ensemble_weights = df_ensemble_weights[s.sort_values(ascending=False).index]

        from tabrepo.plot.plot_ens_weights import create_heatmap
        p = create_heatmap(df=df_ensemble_weights, include_mean=include_mean, **kwargs)
        return p

    # TODO: WIP
    # TODO: Add a non-loo version
    # TODO: Rename
    # FIXME: Make it work with framework_types + max_models_per_type
    def zeroshot_portfolio(
        self,
        configs: list[str] | None = None,
        n_portfolios: int = 200,  # FIXME
        engine: str = "ray",
        rename_columns: bool = True,  # TODO: Align them automatically so this isn't needed
    ) -> pd.DataFrame:
        repo = self.repo

        from scripts.baseline_comparison.baselines import zeroshot_results
        from scripts.baseline_comparison.evaluate_utils import make_scorers

        rank_scorer, normalized_scorer = make_scorers(repo)

        if configs is None:
            configs = repo.configs()

        a = zeroshot_results(
            repo=repo,
            dataset_names=repo.datasets(),
            n_portfolios=[n_portfolios],
            rank_scorer=rank_scorer,
            normalized_scorer=normalized_scorer,
            n_eval_folds=repo.n_folds(),
            configs=configs,
            engine=engine,
        )

        df_zeroshot_portfolio = pd.DataFrame(a)

        if rename_columns:
            df_zeroshot_portfolio = df_zeroshot_portfolio.rename(columns={
                "test_error": "metric_error",
                "method": "framework",
            })
            datasets_info = repo.datasets_info()

            df_zeroshot_portfolio["problem_type"] = df_zeroshot_portfolio["dataset"].map(datasets_info["problem_type"])
            df_zeroshot_portfolio["metric"] = df_zeroshot_portfolio["dataset"].map(datasets_info["metric"])

        return df_zeroshot_portfolio
