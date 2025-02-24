from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer


# TODO: support time limit
# TODO: hyperparameters and good defaults
# TODO: memory usage estimation
# TODO: fix joblib errors when using EBM
# TODO: handle interactions for multiclass
class ExplainableBoostingMachineModel(AbstractModel):
    ag_key = "EBM"
    ag_name = "ExplainableBoostingMachine"

    _category_features: list[str] = None

    def _get_model_type(self):
        match self.problem_type:
            case _ if self.problem_type in (BINARY, MULTICLASS):
                from interpret.glassbox import ExplainableBoostingClassifier

                model_cls = ExplainableBoostingClassifier
            case _ if self.problem_type == REGRESSION:
                from interpret.glassbox import ExplainableBoostingRegressor

                model_cls = ExplainableBoostingRegressor
            case _:
                raise ValueError(f"Unsupported problem type: {self.problem_type}")
        return model_cls

    def _preprocess_nonadaptive(self, X, **kwargs):
        X = super()._preprocess_nonadaptive(X, **kwargs)

        if self._category_features is None:
            self._category_features = list(X.select_dtypes(include="category").columns)

        return X

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        sample_weight: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
        num_cpus: int | str = "auto",
        **kwargs,
    ):
        # Create validation set if not provided to enable early stopping.
        if X_val is None:
            from autogluon.core.utils import generate_train_test_split

            X, X_val, y, y_val = generate_train_test_split(
                X=X,
                y=y,
                problem_type=self.problem_type,
                test_size=0.2,
                random_state=0,
            )

        # Preprocess data.
        X = self.preprocess(X, is_train=True)
        X_val = self.preprocess(X_val, is_train=False)
        paras = self._get_model_params()

        # Handle categorical column types ordinal and nominal columns.
        ordinal_columns = paras.pop("ordinal_columns", [])  # The user can specify ordinal columns.
        nominal_columns = paras.pop("nominal_columns", [])  # The user can specify nominal columns.
        feature_types = []
        for c in self._features:
            if c in ordinal_columns:
                f_type = "ordinal"
            elif c in nominal_columns:
                f_type = "nominal"
            elif c in self._category_features:
                # Fallback if user did not specify column type.
                f_type = "nominal"
            else:
                f_type = "auto"
            feature_types.append(f_type)

        # Default parameters for EBM
        extra_kwargs = dict(
            early_stopping_rounds=50,
            outer_bags=1,  # AutoGluon ensemble creates outer bags, no need for this overhead.
            inner_bags=0,  # We supply the validation set, no need for inner bags.
            objective=get_metric_from_ag_metric(metric=self.stopping_metric, problem_type=self.problem_type),
            feature_names=self._features,
            n_jobs=-1 if isinstance(num_cpus, str) else num_cpus,
        )
        extra_kwargs.update(paras)

        # Init Class
        model_cls = self._get_model_type()
        self.model = model_cls(**extra_kwargs)

        # Handle validation data format for EBM
        fit_X = pd.concat([X, X_val], ignore_index=True)
        fit_y = pd.concat([y, y_val], ignore_index=True)
        bag = np.full(len(fit_X), 1)
        bag[len(X) :] = -1

        # Sample Weights
        fit_sample_weight = np.hstack([sample_weight, sample_weight_val]) if sample_weight is not None else None

        with warnings.catch_warnings():  # try to filter joblib warnings
            warnings.filterwarnings("ignore", category=UserWarning, message=".*resource_tracker: process died.*")
            self.model.fit(fit_X, fit_y, sample_weight=fit_sample_weight, bags=[bag])

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=["int", "float", "category"],
            problem_types=[BINARY, MULTICLASS, REGRESSION],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _more_tags(self) -> dict:
        """EBMs do not yet support refit full."""
        return {"can_refit_full": False}


def get_metric_from_ag_metric(*, metric: Scorer, problem_type: str):
    """Map AutoGluon metric to EBM metric for early stopping."""
    if problem_type == BINARY:
        metric_map = dict(
            log_loss="log_loss",
            accuracy="log_loss",
            roc_auc="log_loss",
            f1="log_loss",
            f1_macro="log_loss",
            f1_micro="log_loss",
            f1_weighted="log_loss",
            balanced_accuracy="log_loss",
            recall="log_loss",
            recall_macro="log_loss",
            recall_micro="log_loss",
            recall_weighted="log_loss",
            precision="log_loss",
            precision_macro="log_loss",
            precision_micro="log_loss",
            precision_weighted="log_loss",
        )
        metric_class = metric_map.get(metric.name, "log_loss")
    elif problem_type == MULTICLASS:
        metric_map = dict(log_loss="log_loss", accuracy="log_loss", roc_auc_ovo_macro="log_loss")
        metric_class = metric_map.get(metric.name, "log_loss")
    elif problem_type == REGRESSION:
        metric_map = dict(
            mean_squared_error="rmse",
            root_mean_squared_error="rmse",
            mean_absolute_error="rmse",
            median_absolute_error="rmse",
            r2="rmse",  # rmse_log maybe?
        )
        metric_class = metric_map.get(metric.name, "rmse")
    else:
        raise AssertionError(f"EBM does not support {problem_type} problem type.")

    return metric_class
