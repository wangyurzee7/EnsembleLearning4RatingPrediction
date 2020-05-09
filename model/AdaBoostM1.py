from sklearn import svm
from sklearn import linear_model
import os
from sklearn.externals import joblib
import numpy as np
import random

from .___init___ import get_model

max_n_dict={
    "SVM":3000,
    # "DecisionTree":2000,
    "DecisionTree":200000,
}

class AdaBoostM1:
    def __init__(self,conf):
        self.inner_model_name=conf["model_params"]["ensemble"]["inner_model"]
        self.max_n=max_n_dict[self.inner_model_name]
        self.task=conf["task"]
        self.t=conf["model_params"]["ensemble"]["t"]
        self.model=[]
        self.conf_bak=conf
    def fit(self,data):
        if self.task=="Classification":
            n_label=max(data["y"])+1
            index_list=[[] for i in range(n_label)]
            for i in range(len(data["y"])):
                index_list[data["y"][i]].append(i)
            n_each=min(self.max_n//n_label, max(map(len,index_list)))
            indice=[]
            for il in index_list:
                indice.extend(il[:n_each])
            if self.conf_bak["shuffle"]:
                random.shuffle(indice)
            data["x"]=data["x"][indice]
            data["y"]=np.array(data["y"])[indice]
        else:
            if len(data["y"])>self.max_n:
                data["x"]=data["x"][:self.max_n]
                data["y"]=data["y"][:self.max_n]
        n=len(data["y"])
        weight=np.array([1/n for i in range(n)])
        self.model.clear()
        beta=[]
        for i in range(self.t):
            curr_model=get_model(self.inner_model_name,self.conf_bak)
            curr_model.fit(data,sample_weight=(n*weight))
            p=curr_model.predict(data)
            if self.task=="Classification":
                eps=np.sum((p!=data["y"])*weight)
            elif self.task=="Regression":
                eps=np.sum((np.abs(p-data["y"])>0.5)*weight)
            if eps>0.5:
                assert i>0
                break
            self.model.append(curr_model)
            curr_beta=eps/(1-eps)
            if eps>0:
                weight=(np.abs(p-data["y"])<=0.5)*curr_beta+(np.abs(p-data["y"])>0.5)
            else:
                print("[ !!!!!!Warning!!!!!! ] eps=0 !!!!!!!!! There may be overfit!")
                eps=1e-7
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
            if self.task=="Classification":
                tmp=[]
                for i in range(max(curr_res)+1):
                    tmp.append(np.sum((curr_res==i)*np.log(1/self.beta)))
                ret.append(np.argmax(tmp))
            elif self.task=="Regression":
                log_beta=np.log(1/self.beta)
                ret.append(np.sum(curr_res*log_beta)/np.sum(log_beta))
        return np.array(ret)
    def dump(self,path):
        joblib.dump(self.model,os.path.join(path,"model.m"))
    def load(self,path):
        self.model=joblib.load(os.path.join(path,"model.m"))
        
