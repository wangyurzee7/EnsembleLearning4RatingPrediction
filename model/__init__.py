from .SVM import SVM
from .DecisionTree import DecisionTree

from .Bagging import Bagging
from .AdaBoostM1 import AdaBoostM1


model_list = {
    "SVM": SVM,
    "DecisionTree": DecisionTree,
    "Bagging": Bagging,
    "AdaBoost.M1": AdaBoostM1,
}


def get_model(model_name,conf):
    if model_name in model_list.keys():
        return model_list[model_name](conf)
    else:
        raise NotImplementedError
