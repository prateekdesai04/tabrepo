from __future__ import annotations

import copy

from autogluon.tabular.register import ag_model_register, ModelRegister

from tabrepo.benchmark.models.ag import ExplainableBoostingMachineModel, RealMLPModel, TabPFNV2Model, TabPFNV2ClientModel, TabDPTModel

tabrepo_model_register: ModelRegister = copy.deepcopy(ag_model_register)

_models_to_add = [
    ExplainableBoostingMachineModel,
    RealMLPModel,
    TabPFNV2Model,
    TabPFNV2ClientModel,
    TabDPTModel,
]

for _model_cls in _models_to_add:
    tabrepo_model_register.add(_model_cls)
