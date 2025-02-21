from __future__ import annotations

from typing import Type

from autogluon.core.models import AbstractModel

from tabrepo.scripts_v5.abstract_class import AbstractExecModel
from tabrepo.scripts_v5.AutoGluon_class import AGSingleWrapper
from tabrepo.benchmark.experiment_runner import ExperimentRunner, OOFExperimentRunner
from tabrepo.utils.cache import AbstractCacheFunction, CacheFunctionDummy
from autogluon_benchmark.tasks.task_wrapper import OpenMLTaskWrapper


class Experiment:
    """

    Parameters
    ----------
    name: str,
    method_cls: Type[AbstractExecModel],
    method_kwargs: dict,
    experiment_cls: Type[ExperimentRunner], default OOFExperimentRunner,

    """
    def __init__(
        self,
        name: str,
        method_cls: Type[AbstractExecModel],
        method_kwargs: dict,
        experiment_cls: Type[ExperimentRunner] = OOFExperimentRunner,
    ):
        self.name = name
        self.method_cls = method_cls
        self.method_kwargs = method_kwargs
        self.experiment_cls = experiment_cls

    def construct_method(self, problem_type: str, eval_metric) -> AbstractExecModel:
        return self.method_cls(
            problem_type=problem_type,
            eval_metric=eval_metric,
            **self.method_kwargs,
        )

    def run(
        self,
        task: OpenMLTaskWrapper | None,
        fold: int,
        task_name: str,
        cacher: AbstractCacheFunction | None = None,
        ignore_cache: bool = False,
    ) -> object:
        if cacher is None:
            cacher = CacheFunctionDummy()
        if task is not None:
            out = cacher.cache(
                fun=self.experiment_cls.init_and_run,
                fun_kwargs=dict(
                    method_cls=self.method_cls,
                    task=task,
                    fold=fold,
                    task_name=task_name,
                    method=self.name,
                    fit_args=self.method_kwargs,
                ),
                ignore_cache=ignore_cache,
            )
        else:
            # load cache, no need to load task
            out = cacher.cache(fun=None, fun_kwargs=None, ignore_cache=ignore_cache)
        return out


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
            experiment_cls=OOFExperimentRunner,
        )
