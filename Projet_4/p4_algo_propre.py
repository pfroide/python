# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 21:12:42 2018

@author: Toni
"""

# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor

import warnings
warnings.filterwarnings("ignore")

#joblib

# Lieu où se trouve le fichier
_DOSSIER = 'C:\\Users\\Toni\\Desktop\\'
_DOSSIERTRAVAIL = 'C:\\Users\\Toni\\python\\python\\Projet_4\\pkl'
_FICHIERDATA = _DOSSIER + 'p4_bdd_clean_toni.csv'

def main():
    """
    Fonction main
    """

    # Récupération des dataset
    data = pd.read_csv(_FICHIERDATA, error_bad_lines=False, low_memory=False)
    del data['Unnamed: 0']
    
    data = data[data['ARR_DELAY'] <= 45]
    
    datanum = data.copy()
    datanum.describe()

    liste = datanum['UNIQUE_CARRIER'].unique()
    
    # Logging for Visual Comparison
    log_cols=["Classifier", "Id", "RMSE", "R2"]
    log = pd.DataFrame(columns=log_cols)
    
    for compagnie in liste[1:2]:
        datanum = data.copy()

        print('\n\nPour la compagnie', compagnie)
        
        datanum = datanum[datanum['UNIQUE_CARRIER']==compagnie]
        del datanum['UNIQUE_CARRIER']
        del datanum['ARR_TIME_BLK']
        del datanum['DISTANCE_GROUP']

        # Transposition en 0 et 1 des valeurs non-numériques
        liste_criteres = ['ORIGIN',
                          'DEP_TIME_BLK',
                          'DAY_OF_WEEK',
                          'DEST',
                          'MONTH']
        # Axe X
        data_x = datanum.copy()
        del data_x['ARR_DELAY']
        
        # Axe Y
        data_y = datanum['ARR_DELAY']
          
        # One-Hot encoding
        data_x = pd.get_dummies(data=data_x, columns=liste_criteres)
        
        # On supprime les nan
        data_x.fillna(0, inplace=True)
        data_y.fillna(0, inplace=True)
        
        # Répartition Train/Test
        data_xtrain, data_xtest, data_ytrain, data_ytest = train_test_split(data_x, data_y, train_size=0.75)

        log = all(data_xtrain, data_xtest, data_ytrain, data_ytest, compagnie, log)
        
        print(log)

def testCV(data_xtrain, data_xtest, data_ytrain, data_ytest):
    
    model = linear_model.SGDRegressor()
    # verbose = 1, scoring = 'neg_mean_squared_error'
    #axe_X = np.arange(0.1, 0.5, 0.1)
    #param_grid = {'n_neighbors':axe_X }
    
    param_grid = [{'alpha' : 10.0**-np.arange(1,7),
                   'l1_ratio':[.05, .15, .5, .7, .9, .95, .99, 1]}]
        
    #dictionnaire = {'C':axe_X, 'epsilon':axe_X }

    score = 'neg_mean_squared_error'

    clf = GridSearchCV(model, param_grid=param_grid, verbose=1, cv=5, scoring=score, refit=True)
    
    clf.fit(data_xtrain, data_ytrain)
    best_params = clf.best_params_
    score = sqrt(abs(clf.best_score_))
    
    score_MSE = []
    
    for a, c in zip(clf.cv_results_['mean_test_score'], clf.cv_results_['params']):
        print("\t%s RMSE %.3s pour %s" % ('\tGRIDSCORES\t', sqrt(abs(a)), c))
        score_MSE.append(sqrt(abs(a)))
    
    print('\n\t\t%s\t%f' % (str(best_params), abs(score)))
    
def all(data_xtrain, data_xtest, data_ytrain, data_ytest, compagnie, log):
        
    from sklearn.externals import joblib

    log_cols=["Classifier", "Id", "RMSE", "R2"]
    score = 'neg_mean_squared_error'
        
    classifiers = [SGDRegressor(),
                   AdaBoostRegressor(),
                   LinearRegression(),
                   ElasticNetCV(),
                   LassoCV(),
                   OrthogonalMatchingPursuitCV(),
                   Ridge()]
    
    for clf in classifiers:
        clf.fit(data_xtrain, data_ytrain)
        name = clf.__class__.__name__
        
        fichier = _DOSSIERTRAVAIL + "\\" + name + "_" + compagnie + ".pkl"
        
        joblib.dump(clf, fichier)
        
        #clf = joblib.load(fichier) 

        print("="*30)
        print(name)
        
        print('****Resultats****')
        train_predictions = clf.predict(data_xtest)
        mse = sqrt(abs(mean_squared_error(data_ytest, train_predictions)))
        r2 = 100 * r2_score(data_ytest, train_predictions)
        
        print("RMSE : ", round(mse, 2))
        print("Log Loss : ", round(r2, 2))
        
        log_entry = pd.DataFrame([[name, compagnie, mse, r2]], columns=log_cols)
        log = log.append(log_entry)
            
        print("="*30)
    
    return log

def test(data_xtrain, data_xtest, data_ytrain, data_ytest):
    
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    
    params={'alpha': [25,10,4,2,1.0,0.8,0.5,0.3,0.2,0.1,0.05,0.02,0.01]}
    rdg_reg = Ridge()
    clf = GridSearchCV(rdg_reg,params,cv=2,verbose = 1, scoring = 'neg_mean_squared_error')
    clf.fit(data_xtrain, data_ytrain)
    train_predictions = clf.predict(data_xtest)
    mse = sqrt(abs(mean_squared_error(data_ytest, train_predictions)))
        
    print(mse)
    
    print(clf.best_params_)
    #{'alpha': 4}
    
    print(pd.DataFrame(clf.cv_results_))
    
    for mean in clf.cv_results_['mean_test_score']:
        print("\t MSE = %0.3f" % (sqrt(abs(mean))))

