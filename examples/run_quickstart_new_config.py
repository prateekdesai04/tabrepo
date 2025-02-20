from __future__ import annotations

import os

import pandas as pd

from tabrepo import EvaluationRepository, EvaluationRepositoryCollection, Evaluator
from tabrepo.benchmark.experiment_constructor import AGModelExperiment
from tabrepo.benchmark.experiment_utils import ExperimentBatchRunner


if __name__ == '__main__':
    # Load Context
    context_name = "D244_F3_C1530_10"  # 10 smallest datasets. To run larger, set to "D244_F3_C1530_200"
    expname = "./experiments/quickstart_new_config"  # folder location to save all experiment artifacts
    ignore_cache = True  # set to True to overwrite existing caches and re-run experiments from scratch

    # The original TabRepo artifacts for the 1530 configs
    repo_og: EvaluationRepository = EvaluationRepository.from_context(context_name, cache=True)
    datasets = repo_og.datasets()

    # Sample for a quick demo
    datasets = datasets[:3]
    folds = [0]

    # To run everything:
    # datasets = repo_og.datasets()
    # folds = repo_og.folds

    # import your model classes (can be custom, must inherit from AbstractModel)
    from autogluon.tabular.models import LGBModel, XGBoostModel

    # This list of methods will be fit sequentially on each task (dataset x fold)
    methods = [
        AGModelExperiment(  # Wrapper for fitting a single model via AutoGluon
            # The name you want the config to have
            name="LightGBM_c1_BAG_L1_Reproduced",

            # The class of the model. Can also be a string if AutoGluon recognizes it, such as `"GBM"`
            # Supports any model that inherits from `autogluon.core.models.AbstractModel`
            model_cls=LGBModel,  # model_cls="GBM",  <- identical
            model_hyperparameters={},  # The non-default model hyperparameters.
            fit_kwargs={"num_bag_folds": 8},  # num_bag_folds=8 was used in the TabRepo 2024 paper
        ),
        AGModelExperiment(
            name="XGBoost_c1_BAG_L1_Reproduced",
            model_cls=XGBoostModel,
            model_hyperparameters={},
            fit_kwargs={"num_bag_folds": 8},
        ),
    ]

    tids = [repo_og.dataset_to_tid(dataset) for dataset in datasets]
    repo: EvaluationRepository = ExperimentBatchRunner().generate_repo_from_experiments(
        expname=expname,
        tids=tids,
        folds=folds,
        methods=methods,
        task_metadata=repo_og.task_metadata,
        ignore_cache=ignore_cache,
        convert_time_infer_s_from_batch_to_sample=True,
    )

    repo.print_info()

    save_path = "repo_quickstart_new_config"
    repo.to_dir(path=save_path)  # Load the repo later via `EvaluationRepository.from_dir(save_path)`

    new_configs = repo.configs()
    print(f"New Configs   : {new_configs}")
    print(f"New Configs Hyperparameters: {repo.configs_hyperparameters()}")

    shared_datasets = [d for d in repo.datasets(union=False) if d in repo_og.datasets()]
    repo_combined = EvaluationRepositoryCollection(repos=[repo_og, repo], config_fallback="ExtraTrees_c1_BAG_L1")
    repo_combined = repo_combined.subset(datasets=shared_datasets)

    repo_combined.print_info()

    comparison_configs_og = [
        "RandomForest_c1_BAG_L1",
        "ExtraTrees_c1_BAG_L1",
        "LightGBM_c1_BAG_L1",
        "XGBoost_c1_BAG_L1",
        "CatBoost_c1_BAG_L1",
        "NeuralNetTorch_c1_BAG_L1",
        "NeuralNetFastAI_c1_BAG_L1",
    ]

    # Include our new configs
    comparison_configs = comparison_configs_og + new_configs

    # Baselines to compare configs with
    # baselines = repo_combined.baselines(union=False)  # to compare with all baselines
    baselines = [
        "AutoGluon_bq_4h8c_2023_11_14",
        "H2OAutoML_4h8c_2023_11_14",
        "autosklearn2_4h8c_2023_11_14",
        "flaml_4h8c_2023_11_14",
        "lightautoml_4h8c_2023_11_14",
    ]

    # create an evaluator to compute comparison metrics such as win-rate and ELO
    evaluator = Evaluator(repo=repo_combined)
    metrics = evaluator.compare_metrics(
        datasets=datasets,
        folds=folds,
        baselines=baselines,
        configs=comparison_configs,
    )

    metrics_tmp = metrics.reset_index(drop=False)

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Config Metrics Example:\n{metrics.head(100)}")

    # Requires `autogluon_bench` and `autogluon_benchmark`
    evaluation_save_dir = os.path.join(expname, "evaluation")
    evaluator_output = evaluator.plot_overall_rank_comparison(
        results_df=metrics,
        save_dir=evaluation_save_dir,
        evaluator_kwargs={"treat_folds_as_datasets": True},
    )
