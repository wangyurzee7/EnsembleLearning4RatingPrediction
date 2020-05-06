from sklearn import svm
from sklearn import linear_model
import os
from sklearn.externals import joblib
import numpy as np

from .___init___ import get_model

class AdaBoostM1:
    def __init__(self,conf):
        self.max_n=5000
        assert conf["task"]=="Classification"
        self.t=conf["model_params"]["ensemble"]["t"]
        self.inner_model_name=conf["model_params"]["ensemble"]["inner_model"]
        self.model=[]
        self.conf_bak=conf
    def fit(self,data):
        if len(data["y"])>self.max_n:
            data["x"]=data["x"][:self.max_n]
            data["y"]=data["y"][:self.max_n]
        n=len(data["y"])
        weight=np.array([1/n for i in range(n)])
        self.model.clear()
        beta=[]
        for i in range(self.t):
            curr_model=get_model(self.inner_model_name,self.conf_bak)
            curr_model.fit(data,sample_weight=weight)
            p=curr_model.predict(data)
            eps=np.sum((p!=data["y"])*weight)
            if eps>0.5:
                assert i>0
                break
            self.model.append(curr_model)
            curr_beta=eps/(1-eps)
            if eps>0:
                weight=(p==data["y"])*curr_beta+(p!=data)
            else:
                print("[ !!!!!!Warning!!!!!! ] eps=0 !!!!!!!!! There may be overfit!")
                weight=np.array([1/n for i in range(n)])
            beta.append(curr_beta)
            weight=weight/np.sum(weight)
        self.beta=np.array(beta)
            
    def predict(self,data):
        result=[]
        for m in self.model:
            result.append(m.predict(data))
        result=np.array(result)
        n=result.shape[1]
        ret=[]
        for i in range(n):
            curr_res=result[:,i]
            tmp=[]
            for i in range(max(curr_res)+1):
                tmp.append(np.sum((curr_res==i)*np.log(1/self.beta)))
            ret.append(np.argmax(tmp))
        return np.array(ret)
    def dump(self,path):
        joblib.dump(self.model,os.path.join(path,"model.m"))
    def load(self,path):
        self.model=joblib.load(os.path.join(path,"model.m"))
        
