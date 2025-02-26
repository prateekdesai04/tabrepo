from __future__ import annotations

import datetime
from typing import Type

import pandas as pd

from autogluon.core.data.label_cleaner import LabelCleaner, LabelCleanerDummy
from autogluon.core.metrics import get_metric, Scorer
from autogluon_benchmark.frameworks.autogluon.run import ag_eval_metric_map
from autogluon_benchmark.tasks.task_wrapper import OpenMLTaskWrapper
from tabrepo.benchmark.models.wrapper.abstract_class import AbstractExecModel


class ExperimentRunner:
    def __init__(
        self,
        method_cls: Type[AbstractExecModel],
        task: OpenMLTaskWrapper,
        fold: int,
        task_name: str,
        method: str,
        fit_args: dict | None = None,
        cleanup: bool = True,
    ):
        self.method_cls = method_cls
        self.task = task
        self.fold = fold
        self.task_name = task_name
        self.method = method
        self.fit_args = fit_args
        self.cleanup = cleanup
        self.eval_metric_name = ag_eval_metric_map[self.task.problem_type]  # FIXME: Don't hardcode eval metric
        self.eval_metric: Scorer = get_metric(metric=self.eval_metric_name, problem_type=self.task.problem_type)
        self.model = None
        self.X, self.y, self.X_test, self.y_test = self.task.get_train_test_split(fold=self.fold)
        self.label_cleaner = LabelCleaner.construct(problem_type=self.task.problem_type, y=self.y)

    def init_method(self) -> AbstractExecModel:
        model = self.method_cls(
            problem_type=self.task.problem_type,
            eval_metric=self.eval_metric,
            **self.fit_args,
        )
        return model

    def run_model_fit(self) -> dict:
        return self.model.fit_custom(X=self.X, y=self.y, X_test=self.X_test)

    def run(self):
        out = self._run()
        if self.cleanup:
            self._cleanup()
        return out

    @classmethod
    def init_and_run(
        cls,
        method_cls: Type[AbstractExecModel],
        task: OpenMLTaskWrapper,
        fold: int,
        task_name: str,
        method: str,
        fit_args: dict = None,
        cleanup: bool = True,
    ):
        obj = cls(
            method_cls=method_cls,
            task=task,
            fold=fold,
            task_name=task_name,
            method=method,
            fit_args=fit_args,
            cleanup=cleanup,
        )
        return obj.run()

    def _run(self):
        utc_time = datetime.datetime.now(datetime.timezone.utc)
        time_start_str = utc_time.strftime('%Y-%m-%d %H:%M:%S')
        time_start = utc_time.timestamp()
        self.model = self.init_method()
        out = self.run_model_fit()
        out = self.post_fit(out=out)
        out["metric_error"] = self.evaluate(
            y_true=self.y_test,
            y_pred=out["predictions"],
            y_pred_proba=out["probabilities"],
        )
        out = self.post_evaluate(out=out)
        out["experiment_metadata"] = self._experiment_metadata(time_start=time_start, time_start_str=time_start_str)
        out = self.convert_to_output(out=out)
        return out

    def post_fit(self, out: dict) -> dict:
        return out

    def post_evaluate(self, out: dict) -> dict:
        out["framework"] = self.method
        out["dataset"] = self.task_name
        out["tid"] = self.task.task_id
        out["fold"] = self.fold
        out["problem_type"] = self.task.problem_type
        out["metric"] = self.eval_metric_name

        if self.model.can_get_oof:
            simulation_artifact = self.model.get_oof()
            simulation_artifacts = {self.task_name: {self.fold: simulation_artifact}}
        else:
            simulation_artifacts = None
        out["simulation_artifacts"] = simulation_artifacts
        if hasattr(self.model, "get_metadata"):
            out["method_metadata"] = self.model.get_metadata()
        return out

    def _experiment_metadata(self, time_start: float, time_start_str: str) -> dict:
        metadata = {}
        metadata["experiment_cls"] = self.__class__.__name__
        metadata["method_cls"] = self.method_cls.__name__
        time_end = datetime.datetime.now(datetime.timezone.utc).timestamp()
        metadata["time_start"] = time_start
        metadata["time_end"] = time_end
        metadata["total_duration"] = time_end - time_start
        metadata["time_start_str"] = time_start_str
        return metadata

    # FIXME: outdated
    def convert_to_output(self, out: dict):
        out.pop("predictions")
        out.pop("probabilities")
        out.pop("simulation_artifacts")
        out.pop("experiment_metadata", None)
        out.pop("method_metadata", None)

        df_results = pd.DataFrame([out])
        ordered_columns = ["dataset", "fold", "framework", "metric_error", "metric", "time_train_s"]
        columns_reorder = ordered_columns + [c for c in df_results.columns if c not in ordered_columns]
        df_results = df_results[columns_reorder]

        return df_results

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_pred_proba: pd.Series | pd.DataFrame | None,
    ) -> float:
        return evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            scorer=self.eval_metric,
            label_cleaner=self.label_cleaner,
            problem_type=self.task.problem_type,
        )

    def _cleanup(self):
        self.model.cleanup()


class OOFExperimentRunner(ExperimentRunner):
    def post_evaluate(self, out: dict) -> dict:
        out = super().post_evaluate(out=out)
        if self.model.can_get_oof:
            simulation_artifact = self.model.get_oof()
            if self.task.problem_type == "regression":
                simulation_artifact["pred_proba_dict_test"] = self.label_cleaner.transform(out["predictions"])
            else:
                simulation_artifact["pred_proba_dict_test"] = self.label_cleaner.transform_proba(out["probabilities"], as_pandas=True)
                if self.task.problem_type == "binary":
                    simulation_artifact["pred_proba_dict_test"] = simulation_artifact["pred_proba_dict_test"].iloc[:, 1]
            simulation_artifact["y_test"] = self.label_cleaner.transform(self.y_test)
            simulation_artifact["label"] = self.task.label
            simulation_artifact["metric"] = self.eval_metric_name

            out["metric_error_val"] = self.model.get_metric_error_val()
            # out["metric_error_val"] = evaluate(
            #     y_true=simulation_artifact["y_val"],
            #     y_pred=self.label_cleaner.transform(out["predictions"]),
            #     y_pred_proba=self.label_cleaner.transform_proba(out["probabilities"])
            # )
            # out["metric_error_val"] = self.eval_metric.error(simulation_artifact["y_val"], simulation_artifact["pred_proba_dict_val"])

            simulation_artifact["pred_proba_dict_val"] = {self.method: simulation_artifact["pred_proba_dict_val"]}
            simulation_artifact["pred_proba_dict_test"] = {self.method: simulation_artifact["pred_proba_dict_test"]}
            simulation_artifacts = {self.task_name: {self.fold: simulation_artifact}}
        else:
            simulation_artifacts = None
        out["simulation_artifacts"] = simulation_artifacts
        return out

    def convert_to_output(self, out: dict) -> dict:
        ignored_columns = [
            "predictions",
            "probabilities",
            "simulation_artifacts",
            "experiment_metadata",
            "method_metadata",
        ]
        out_keys = list(out.keys())

        ordered_columns = ["dataset", "fold", "framework", "metric_error", "metric", "time_train_s"]
        columns_reorder = ordered_columns + [c for c in out_keys if c not in ordered_columns and c not in ignored_columns]
        df_results = pd.DataFrame([{k: out[k] for k in columns_reorder}])
        df_results = df_results[columns_reorder]

        out["df_results"] = df_results
        out.pop("predictions")
        out.pop("probabilities")

        return out


def evaluate(y_true: pd.Series, y_pred: pd.Series, y_pred_proba: pd.DataFrame, scorer: Scorer, problem_type: str, label_cleaner: LabelCleaner = None) -> float:
    if label_cleaner is None:
        label_cleaner = LabelCleanerDummy(problem_type=problem_type)
    y_true = label_cleaner.transform(y_true)
    if scorer.needs_pred:
        y_pred = label_cleaner.transform(y_pred)
        error = scorer.error(y_true=y_true, y_pred=y_pred)
    elif problem_type == "binary":
        y_pred_proba = label_cleaner.transform_proba(y_pred_proba, as_pandas=True)
        error = scorer.error(y_true=y_true, y_pred=pd.DataFrame(y_pred_proba).iloc[:, 1])
    else:
        y_pred_proba = label_cleaner.transform_proba(y_pred_proba, as_pandas=True)
        error = scorer.error(y_true=y_true, y_pred=y_pred_proba)
    return error
