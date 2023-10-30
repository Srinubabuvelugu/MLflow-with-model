import pandas as pd 
import numpy as np
import os
import mlflow
import argparse
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC,LinearSVC

from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error,r2_score,roc_auc_score
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")
## To read the data set from local
@dataclass
class data_path_congfig:
    arti:str = os.path.dirname("artifacts")
    path:str = os.path.join("notebook","wine_quality.csv")
    train_data_path:str = os.path.join("artifacts","train_data.csv")
    test_data_path:str = os.path.join("artifacts","test_data.csv")
    arti:str = os.path.dirname("artifacts")

## evaluate function creation
def evaluate(model, X_trian, X_test, y_train, y_test):
    ## Model training
    model.fit(X_trian,y_train)
    ## Model Prediction
    y_pred = model.predict(X_test)
    ## Error values 
    """ mse = mean_squared_error(y_test,y_pred)
    rmse = mse**0.5
    mae = mean_absolute_error(y_pred,y_test)
    score = r2_score(y_test,y_pred)
    dic = {
        "model":model,
        "MSE":mse,
        "RMSE":rmse,
        "MAE":mae,
        "R2_score":score
    }
    return dic"""
    mlflow.sklearn.log_model(model,"RandomForestClassifier")
    score = accuracy_score(y_test,y_pred)
    pred_proba = model.predict_proba(X_test)
    roc = roc_auc_score(y_test,pred_proba,multi_class="ovr")
    return score,roc
class Data_Ingestion:
    def __init__(self,n_estimators,max_depth,max_leaf_nodes):
        self.data_path = data_path_congfig()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes

    def read_data(self):
        try:
            data = pd.read_csv(self.data_path.path)
            return data
        except Exception as e:
            print(e) 

    def get_data(self):
        data=self.read_data() 
        #print(data.head())
    
    def accuracy(self):
        data = self.read_data()
        try:
            
            ## Train_test_split
            train,test = train_test_split(data,test_size=0.25, random_state=11)
            if not os.path.exists("artifacts"):
                os.makedirs("artifacts")
            train.to_csv(self.data_path.train_data_path,index=False,header =True)
            test.to_csv(self.data_path.test_data_path,index=False,header = True)
            X_train = train.drop("quality",axis=1)
            X_test = test.drop("quality",axis=1)
            y_train = train["quality"]
            y_test = test["quality"]

            ## model bulding
            with mlflow.start_run():
                
                mlflow.log_params({"n_estimators":self.n_estimators,
                                   "max_depth":self.max_depth,
                                   "max_leaf_nodes":self.max_leaf_nodes
                                   })
                
                
                #models = [ElasticNet(),RandomForestClassifier(),DecisionTreeClassifier(),LogisticRegression(),SVC(),LinearSVC()]
                models = [RandomForestClassifier(n_estimators=self.n_estimators,max_depth=self.max_depth,max_leaf_nodes=self.max_leaf_nodes)]
                ## Evalute function for r2_score, mean_squared_error, mean_absulote_error
                for model in models:
                    res,roc = evaluate(model,X_train,X_test, y_train, y_test)
                    print(f"accuracy score:{res} and roc_auc_score: {roc}")
                    mlflow.log_metrics({"accuracy score":res,
                                   "roc_auc_score":roc
                                   })
                    if not os.path.exists("outputs"):
                        os.makedirs("outputs")
                    if os.path.exists("outputs"):
                        with open("outputs/test.txt", "a+") as f:
                            f.write(f"n_estimators= {self.n_estimators},max_depth = {self.max_depth},max_leaf_nodes = {self.max_leaf_nodes} ====>> accuracy score:{res} and roc_auc_score: {roc} \n")
                    mlflow.log_artifacts("outputs")
                                    
                                   
        except Exception as e:
            print(e)
     

if __name__ == "__main__":

    ## creating argument parameters
    args = argparse.ArgumentParser()
    args.add_argument("--n_estimators","-n",default=50,type=int)
    args.add_argument("--max_depth","-md",default=8,type=int)
    args.add_argument("--max_leaf_nodes","-ml",default=2,type=int)
    parse_args = args.parse_args()
    try:
        obj = Data_Ingestion(parse_args.n_estimators,parse_args.max_depth, parse_args.max_leaf_nodes)
        obj.get_data()
        obj.accuracy()
    except Exception as e:
        print(e)
