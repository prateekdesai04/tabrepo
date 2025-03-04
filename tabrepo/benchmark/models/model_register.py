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


def infer_model_cls(model_cls: str, model_register: ModelRegister = None):
    if model_register is None:
        model_register = tabrepo_model_register
    if isinstance(model_cls, str):
        if model_cls in model_register.key_to_cls_map():
            model_cls = model_register.key_to_cls(key=model_cls)
        elif model_cls in model_register.name_map().values():
            for real_model_cls in model_register.model_cls_list:
                if real_model_cls.ag_name == model_cls:
                    model_cls = real_model_cls
                    break
        elif model_cls in [str(real_model_cls.__name__) for real_model_cls in model_register.model_cls_list]:
            for real_model_cls in model_register.model_cls_list:
                if model_cls == str(real_model_cls.__name__):
                    model_cls = real_model_cls
                    break
        else:
            raise AssertionError(f"Unknown model_cls: {model_cls}")
    return model_cls
