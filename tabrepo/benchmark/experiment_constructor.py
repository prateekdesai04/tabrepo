from __future__ import annotations

from typing import Type

from autogluon.core.models import AbstractModel

from tabrepo.scripts_v5.abstract_class import AbstractExecModel
from tabrepo.scripts_v5.AutoGluon_class import AGSingleWrapper


class Experiment:
    def __init__(
        self,
        name: str,
        method_cls: Type[AbstractExecModel],
        method_kwargs: dict,
    ):
        self.name = name
        self.method_cls = method_cls
        self.method_kwargs = method_kwargs

    def construct_method(self, problem_type: str, eval_metric) -> AbstractExecModel:
        return self.method_cls(
            problem_type=problem_type,
            eval_metric=eval_metric,
            **self.method_kwargs,
        )


# convenience wrapper
class AGModelExperiment(Experiment):
    def __init__(
        self,
        name: str,
        model_cls: Type[AbstractModel],
        model_hyperparameters: dict,
        **method_kwargs,
    ):
        super().__init__(
            name=name,
            method_cls=AGSingleWrapper,
            method_kwargs={
                "model_cls": model_cls,
                "model_hyperparameters": model_hyperparameters,
                **method_kwargs,
            },
        )
