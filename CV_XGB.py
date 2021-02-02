import numpy as np
import random
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn import metrics
import xgboost as xgb
import os
import warnings
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle  
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from numpy import interp
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


    
    
def modelfit(alg, x_t, y_t,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(x_t, label=y_t)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(x_t, y_t,eval_metric='auc')

    #Predict training set:
    dtrain_predictions = alg.predict(x_t)
    dtrain_predprob = alg.predict_proba(x_t)[:,1]

    #Print model report:
    print('Best number of trees = {}'.format(cvresult.shape[0]))
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y_t, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_t, dtrain_predprob))

    return cvresult.shape[0]
    
    
    
def CV_XGB(x_train_XGB,y_train_XGB):
    clf = xgb.XGBClassifier(
                learning_rate=0.1,
                n_estimators=100,
                max_depth=6,
                min_child_weight=2,
                #gamma=4,
                seed=0)
                
    ###交叉验证后最好的树
    # print('Best number of trees = {}'.format(cvresult.shape[0]))
    # clf.set_params(n_estimators=cvresult.shape[0])#把clf的参数设置成最好的树对应的参数
    # clf.fit(x_train_XGB,y_train_XGB, eval_metric='auc')#训练clf
    ###pred = clf.predict_proba(X_test, ntree_limit=cvresult.shape[0])
    ###Y_predict=clf.predict(x_test_XGB)
    # pred = clf.predict_proba(x_test_XGB, ntree_limit=cvresult.shape[0])
    # Y_predict = pred[:,1]
    
    
    
    # n_estimators_t=modelfit(clf,x_train_XGB,y_train_XGB)
    n_estimators_t=100
    
    param_test1 = {
    'max_depth':range(6,8,1),#6
    'min_child_weight':range(1,4,1),#2
    }
    gsearch1 = GridSearchCV(
        estimator = clf,
        param_grid = param_test1,scoring ='roc_auc',cv = 10)# iid = False,n_jobs = 4,
    gsearch1.fit(x_train_XGB,y_train_XGB)
    print(gsearch1.best_params_,gsearch1.best_score_)
    # max_depth_t=gsearch1.best_params_['max_depth']
    # min_child_weight_t = gsearch1.best_params_['max_depth']
    
    clf = xgb.XGBClassifier(
                learning_rate=0.1,
                n_estimators=n_estimators_t,
                max_depth=gsearch1.best_params_['max_depth'],
                min_child_weight=gsearch1.best_params_['min_child_weight'],
                seed=0)
                
                
    param_test2 = {
    'gamma':[0,0.2,0.4,0.6,0.8,1]
    }
    gsearch2 = GridSearchCV(
        estimator = clf,
        param_grid = param_test2,scoring ='roc_auc',cv = 10)# iid = False,n_jobs = 4,
    gsearch2.fit(x_train_XGB,y_train_XGB)
    print(gsearch2.best_params_,gsearch2.best_score_)
    
    clf = xgb.XGBClassifier(
                learning_rate=0.1,
                n_estimators=n_estimators_t,
                nthread=8,
                max_depth=gsearch1.best_params_['max_depth'],
                min_child_weight=gsearch1.best_params_['min_child_weight'],
                gamma=gsearch2.best_params_['gamma'],
                seed=0)
    param_test3 = {
    'subsample':[0.8,0.9,1],#0.9,0.7,0.9,0.8,0.8
    'colsample_bytree':[0.8,0.9,1]#1,0.8,0.9,0.85,0.9
    }
    gsearch3 = GridSearchCV(
        estimator = clf,
        param_grid = param_test3,scoring ='roc_auc',cv = 10)
    gsearch3.fit(x_train_XGB,y_train_XGB)
    print(gsearch3.best_params_,gsearch3.best_score_)
    
    # clf = xgb.XGBClassifier(
                # learning_rate=0.1,
                # n_estimators=n_estimators_t,
                # nthread=8,
                # max_depth=gsearch1.best_params_['max_depth'],
                # min_child_weight=gsearch1.best_params_['min_child_weight'],
                # gamma=gsearch2.best_params_['gamma'],
                # subsample=gsearch3.best_params_['subsample'],
                # colsample_bytree=gsearch3.best_params_['colsample_bytree'],
                # seed=0)
    # param_test4 = {
    # 'reg_alpha':[0, 0.001, 0.01]
    # }
    # gsearch4 = GridSearchCV(
        # estimator = clf,
        # param_grid = param_test4,scoring ='roc_auc',cv = 10)
    # gsearch4.fit(x_train_XGB,y_train_XGB)
    # print(gsearch4.best_params_,gsearch4.best_score_)

    
    # params = {
            # n_estimators=n_estimators_t,
            # nthread=8,
            # max_depth=gsearch1.best_params_['max_depth'],
            # min_child_weight=gsearch1.best_params_['min_child_weight'],
            # gamma=gsearch2.best_params_['gamma'],
            # subsample=gsearch3.best_params_['subsample'],
            # colsample_bytree=gsearch3.best_params_['colsample_bytree'],
            # seed=0
    # }
    params = {
            'learning_rate':0.1,
            'n_estimators':n_estimators_t,
            'max_depth':gsearch1.best_params_['max_depth'],
            'min_child_weight':gsearch1.best_params_['min_child_weight'],
            'gamma':gsearch2.best_params_['gamma'],
            'subsample':gsearch3.best_params_['subsample'],
            'colsample_bytree':gsearch3.best_params_['colsample_bytree'],
            # 'reg_alpha':gsearch4.best_params_['reg_alpha'],
            'seed':0
    }
    return params









