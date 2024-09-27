from __future__ import annotations

import pandas as pd

from tabrepo import load_repository, EvaluationRepository
from TabPFN_class import CustomTabPFN
from TabPFNv2_class import CustomTabPFNv2
from experiment_utils import run_experiments, convert_leaderboard_to_configs

if __name__ == '__main__':
    # Load Context
    context_name = "D244_F3_C1530_30"
    repo: EvaluationRepository = load_repository(context_name, cache=True)

    expname = "./initial_experiment_tabpfn_v5"  # folder location of all experiment artifacts
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    datasets = [
        "blood-transfusion-service-center",
        "Australian",
    ]

    # Add a check here if the dataset belong to repo
    tids = [repo.dataset_to_tid(dataset) for dataset in datasets]

    methods_dict_tabpfn = {"TABPFNv2": {}}
    methods_tabpfn = list(methods_dict_tabpfn.keys())

    results_lst = run_experiments(
        expname=expname,
        tids=tids,
        folds=repo.folds,
        methods=methods_tabpfn,
        methods_dict=methods_dict_tabpfn,
        method_cls=CustomTabPFNv2,
        task_metadata=repo.task_metadata,
        ignore_cache=ignore_cache,
    )

    results_df = pd.concat(results_lst, ignore_index=True)

    results_df = convert_leaderboard_to_configs(results_df)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(results_df)

    comparison_configs = [
        "RandomForest_c1_BAG_L1",
        "ExtraTrees_c1_BAG_L1",
        "LightGBM_c1_BAG_L1",
        "XGBoost_c1_BAG_L1",
        "CatBoost_c1_BAG_L1",
        "TabPFN_c1_BAG_L1",
        "NeuralNetTorch_c1_BAG_L1",
        "NeuralNetFastAI_c1_BAG_L1",
    ]

    baselines = [
        "AutoGluon_bq_4h8c_2023_11_14",
    ]

    metrics = repo.compare_metrics(
        results_df,
        datasets=datasets,
        folds=repo.folds,
        baselines=baselines,
        configs=comparison_configs,
    )
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Config Metrics Example:\n{metrics}")
    evaluator_output = repo.plot_overall_rank_comparison(
        results_df=metrics,
        save_dir=expname,
    )
