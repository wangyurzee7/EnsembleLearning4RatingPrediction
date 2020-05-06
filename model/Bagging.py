from sklearn import svm
from sklearn import linear_model
import os
from sklearn.externals import joblib
import numpy as np

from .___init___ import get_model

class Bagging:
    def __init__(self,conf):
        self.task=conf["task"]
        self.t=conf["model_params"]["ensemble"]["t"]
        self.m=conf["model_params"]["ensemble"]["m"]
        self.model=[]
        for i in range(self.t):
            self.model.append(get_model(conf["model_params"]["ensemble"]["inner_model"],conf))
    def fit(self,data):
        n=len(data["y"])
        for i in range(self.t):
            indice=np.random.choice(a=n, size=self.m, replace=False, p=None)
            x=data["x"][indice]
            y=np.array(data["y"])[indice]
            self.model[i].fit({"x":x,"y":y})
    def predict(self,data):
        result=[]
        for i in range(self.t):
            result.append(self.model[i].predict(data))
        result=np.array(result)
        n=result.shape[1]
        ret=[]
        for i in range(n):
            if self.task=="Classification":
                ret.append(np.argmax(np.bincount(result[:,i])))
            elif self.task=="Regression":
                ret.append(np.mean(result[:,i]))
        return np.array(ret)
    def dump(self,path):
        joblib.dump(self.model,os.path.join(path,"model.m"))
    def load(self,path):
        self.model=joblib.load(os.path.join(path,"model.m"))
        
