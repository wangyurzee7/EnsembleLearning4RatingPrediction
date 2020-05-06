from sklearn import tree
import os
from sklearn.externals import joblib

def set_model_params(model,params):
    for key in params.keys():
        val=params[key]
        if key=="class_weight":
            model.set_params(class_weight=val)
        else:
            pass

class DecisionTree:
    def __init__(self,conf):
        self.max_n=10000
        if conf["task"]=="Classification":
            self.model=tree.DecisionTreeClassifier()
        elif conf["task"]=="Regression":
            self.model=tree.DecisionTreeRegressor()
        else:
            pass
        set_model_params(self.model,conf["model_params"])
    def fit(self,data,sample_weight=None):
        if len(data["y"])>self.max_n:
            data["x"]=data["x"][:self.max_n]
            data["y"]=data["y"][:self.max_n]
        self.model.fit(data["x"],data["y"],sample_weight=sample_weight)
    def predict(self,data):
        return self.model.predict(data["x"])
    def dump(self,path):
        joblib.dump(self.model,os.path.join(path,"model.m"))
    def load(self,path):
        self.model=joblib.load(os.path.join(path,"model.m"))
        
