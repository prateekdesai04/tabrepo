from __future__ import annotations

import copy
from typing import Type

from autogluon.core.models import AbstractModel

from tabrepo.benchmark.models.wrapper.abstract_class import AbstractExecModel
from tabrepo.benchmark.models.wrapper.AutoGluon_class import AGSingleWrapper
from tabrepo.benchmark.experiment.experiment_runner import ExperimentRunner, OOFExperimentRunner
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


# convenience wrapper
class AGModelBagExperiment(AGModelExperiment):
    """
    AutoGluon Bagged Model Wrapper.

    Will fit the model with `num_bag_folds` folds and `num_bag_sets` sets (aka repeats).
    In total will fit `num_bag_folds * num_bag_sets` models in the bag.
    """
    def __init__(
        self,
        name: str,
        model_cls: Type[AbstractModel],
        model_hyperparameters: dict,
        num_bag_folds: int = 8,
        num_bag_sets: int = 1,
        **method_kwargs,
    ):
        assert isinstance(num_bag_folds, int)
        assert isinstance(num_bag_sets, int)
        assert num_bag_folds >= 2
        assert num_bag_sets >= 1
        if "fit_kwargs" in method_kwargs:
            assert "num_bag_folds" not in method_kwargs["fit_kwargs"], f"Set `num_bag_folds` directly in `AGModelBagExperiment` rather than in `fit_kwargs`"
            assert "num_bag_sets" not in method_kwargs["fit_kwargs"], f"Set `num_bag_sets` directly in `AGModelBagExperiment` rather than in `fit_kwargs`"
            method_kwargs["fit_kwargs"] = copy.deepcopy(method_kwargs["fit_kwargs"])
        else:
            method_kwargs["fit_kwargs"] = {}
        method_kwargs["fit_kwargs"]["num_bag_folds"] = num_bag_folds
        method_kwargs["fit_kwargs"]["num_bag_sets"] = num_bag_sets
        super().__init__(
            name=name,
            model_cls=model_cls,
            model_hyperparameters=model_hyperparameters,
            **method_kwargs,
        )
