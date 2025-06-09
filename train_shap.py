import numpy as np 
import pandas as pd 
import shap
import os

from sklearn.model_selection import (KFold, StratifiedKFold, StratifiedGroupKFold,
                                     GridSearchCV, train_test_split)

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler,EditedNearestNeighbours

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def machine_learning_exp(model,X,Y):
    
    # Number of folds for the outer cross-validation
    OUTER_K = 5
    # Number of folds for the inner cross-validation
    INNER_K = 3
    SEED = 22

    # Standard machine learning parameters
    MAX_ITER = 250000  # for support vector classifier
    C_LIST = [1e-3, 1e-2, 1e-1, 1e0] # parameters for SVM
    N_NEIGHBORS_LIST = list(range(1, 10)) # NUmber of neighbors to consider if you use KNN algorithm 


    all_y_true = []
    all_x_test = []
    all_y_pred = []
    accuracies = []
    additional_metrics = []
    all_hps = []
    auc = []
    all_shap_values = []
    
  

    # The split of the fold for the crossvalidation keeps the original rate beteen postive and negative instances 
    out_kf = StratifiedKFold(n_splits=OUTER_K,shuffle=True,random_state=SEED)
    in_kf = StratifiedKFold(n_splits=INNER_K,shuffle=True,random_state=SEED)

    out_split = out_kf.split(X, Y)

    print("out_split",out_split)

    for k, out_idx in enumerate(out_split):
        
        print("type out_idx",type(out_idx))
        print(f'Outer fold {k+1}/{OUTER_K}')
        X_train, X_test = X.iloc[out_idx[0]], X.iloc[out_idx[1]]
        Y_train, Y_test = Y[out_idx[0]], Y[out_idx[1]]
        
        
        X_train, Y_train = shuffle(X_train, Y_train, random_state=SEED)

        
        
        all_y_true.extend(Y_test)

        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        
        
        
        print( "-------------------------------------")
      
        #Normalize the data only using the training set
        
        maxs = X_train.max(axis =0)
        mins = X_train.min(axis =0)

        
        X_train = (X_train - mins)/(maxs-mins)
        X_test = (X_test - mins)/(maxs-mins)
        
        #Oversample the minority class
        X_train, Y_train = ADASYN(random_state=SEED).fit_resample(X_train,Y_train)

        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test= pd.DataFrame(X_test, columns=X.columns)
        Y_train = pd.Series(Y_train)
        
        print( "-------------------------------------")

        in_split = in_kf.split(X_train, Y_train)

        if model == 'lda':
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_train, Y_train)
            y_pred = lda.predict(X_test).tolist()
            all_hps.append(None)
            
            explainer = shap.LinearExplainer(lda, X_train)
            shap_values = explainer.shap_values(X_test)
            
            print(f'MODEL: {model}')
           
            print(f"Y test labels: {Y_test.tolist()}")
            
            print("The shap values from each test instance are  being multiplied by the corresponding label (+1 or -1).")
            shap_values = shap_values * Y_test.reshape(-1, 1)
            all_shap_values.append(shap_values)
            
            all_x_test.append(X_test)

            
        elif model=='knn':

            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()
            
            parameters = {'n_neighbors': N_NEIGHBORS_LIST}
            knn = KNeighborsClassifier()
            clf = GridSearchCV(knn, parameters,scoring = 'accuracy', cv=in_split)
            clf.fit(X_train,Y_train)
            best_knn = clf.best_estimator_
            y_pred = clf.predict(X_test).tolist()
            
            all_hps.append(clf.best_params_['n_neighbors'])

            all_x_test.append(X_test)

            
        elif model=='svc':
            parameters = {'C': C_LIST}
            svc = LinearSVC(max_iter=MAX_ITER, dual='auto')
            clf = GridSearchCV(svc, parameters,scoring = 'accuracy', cv=in_split)
            clf.fit(X_train,Y_train)
            best_svc= clf.best_estimator_
            y_pred = clf.predict(X_test).tolist()
            all_hps.append(clf.best_params_['C'])

            # Initialize the SHAP Linear Explainer (corrected parameter)
            explainer = shap.LinearExplainer(best_svc, X_train)
         
            
            shap_values = explainer.shap_values(X_test)
            
            
            print(f'MODEL: {model}')
            print(f"Y test labels: {Y_test.tolist()}")
            print("The shap values from each test instance are  being multiplied by the corresponding label (+1 or -1).")
            shap_values = shap_values * Y_test.reshape(-1, 1)
            all_shap_values.append(shap_values)
            
            all_x_test.append(X_test)

    
        # Metrics
        accuracies.append(accuracy_score(Y_test, y_pred))
        auc.append(roc_auc_score(Y_test, y_pred))
        prfs = precision_recall_fscore_support(Y_test, y_pred)
        additional_metrics.append(prfs)
        all_y_pred += y_pred

    


    return all_shap_values, all_hps, auc, all_y_true




