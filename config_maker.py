import json
import copy

bagging_param_list=[
    {"t":10,"m":1000},
    {"t":25,"m":1000},
    {"t":50,"m":1000},
    {"t":10,"m":3000},
    {"t":25,"m":3000},
    {"t":50,"m":3000},
]

adaboostm1_param_list=[
    {"t":5},
    {"t":10},
    {"t":15},
    {"t":20},
    {"t":25},
    {"t":30},
]

if __name__=="__main__":
    f_run_sh=open("run_all.sh","w")
    f_cat_sh=open("cat_all.sh","w")
    f_upload_sh=open("upload_all.sh","w")
    for _type in ["clas","regr"]:
        template=json.load(open("config/rating/test_bagging_{}.json".format(_type),"r"))
        for model in ["SVM","DecisionTree"]:
            for param in bagging_param_list:
                conf=copy.deepcopy(template)
                model_name="bagging_{}_model={}_t={}_m={}".format(_type,model,param["t"],param["m"])

                conf["model_name"]=model_name
                conf["model_params"]["ensemble"]["inner_model"]=model
                for k in param.keys():
                    conf["model_params"]["ensemble"][k]=param[k]
                
                file_name="config/rating/release/{}.json".format(model_name)
                json.dump(conf,open(file_name,"w"))

                log_file="models/{}/log.txt".format(model_name)
                csv_file="models/{}/CsvFormatOutputDumper.csv".format(model_name)

                f_run_sh.write("python3 main.py --config {} \n".format(file_name))
                f_cat_sh.write("cat {} \n".format(log_file))
                f_cat_sh.write("echo '' \n")
                f_upload_sh.write("kaggle competitions submit -c thuml2020 -f {} -m \"{}\" \n".format(csv_file,model_name))
    
    # adaboostm1
    for _type in ["clas","regr"]:
        template=json.load(open("config/rating/test_adaboostm1_{}.json".format(_type),"r"))
        for model in ["SVM","DecisionTree"]:
            for param in adaboostm1_param_list:
                conf=copy.deepcopy(template)
                model_name="adaboostm1_{}_model={}_t={}".format(_type,model,param["t"])

                conf["model_name"]=model_name
                conf["model_params"]["ensemble"]["inner_model"]=model
                for k in param.keys():
                    conf["model_params"]["ensemble"][k]=param[k]
                
                file_name="config/rating/release/{}.json".format(model_name)
                json.dump(conf,open(file_name,"w"),indent=2,ensure_ascii=True)

                log_file="models/{}/log.txt".format(model_name)
                csv_file="models/{}/CsvFormatOutputDumper.csv".format(model_name)

                f_run_sh.write("python3 main.py --config {} \n".format(file_name))
                f_cat_sh.write("cat {} \n".format(log_file))
                f_cat_sh.write("echo '' \n")
                f_upload_sh.write("kaggle competitions submit -c thuml2020 -f {} -m \"{}\" \n".format(csv_file,model_name))

    f_run_sh.close()
    f_cat_sh.close()
    f_upload_sh.close()