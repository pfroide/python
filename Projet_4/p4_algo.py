# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 21:12:42 2018

@author: Toni
"""

# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm as cm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LarsCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
    
import warnings
warnings.filterwarnings("ignore")

#joblib

# Lieu où se trouve le fichier
_DOSSIER = 'C:\\Users\\Toni\\Desktop\\'
_DOSSIERTRAVAIL = 'C:\\Users\\Toni\\python\\python\\Projet_4\\images'
_FICHIERDATA = _DOSSIER + 'p4_bdd_clean_toni.csv'

def main():
    """
    Fonction main
    """

    # Récupération des dataset
    data = pd.read_csv(_FICHIERDATA, error_bad_lines=False, low_memory=False)
    del data['Unnamed: 0']
    
    data = data[data['ARR_DELAY'] <= 45]
    
    #del data['CLASSE_DELAY']
    
    # Tests
    #del data['DEST']
    #del data['ORIGIN']
    
    datanum = data.copy()
    datanum.describe()

    liste = datanum['UNIQUE_CARRIER'].unique()
    
    # Logging for Visual Comparison
    log_cols=["Classifier", "Id", "RMSE", "R2"]
    log = pd.DataFrame(columns=log_cols)
    
    for compagnie in liste:
        datanum = data.copy()

        print('\n\nPour la compagnie', compagnie)
        
        datanum = datanum[datanum['UNIQUE_CARRIER']==compagnie]
        del datanum['UNIQUE_CARRIER']
        del datanum['ARR_TIME_BLK']

        # Transposition en 0 et 1 des valeurs non-numériques
        liste_criteres = ['ORIGIN',
                          'DEP_TIME_BLK',
                          'DAY_OF_WEEK',
                          'DEST',
                          'MONTH']

#        liste_criteres = ['DEST', 
#                          'ORIGIN',
#                          'DEP_TIME_BLK',
#                          'ARR_TIME_BLK',
#                          'DISTANCE_GROUP',
#                          'DAY_OF_WEEK', 
#                          'MONTH']
        
    # tester sans ARR_TIME_BLK
    # tester sans les critères d'arrivées
    
    #    liste_criteres = ['DEST',
    #                      'ORIGIN',
    #                      'UNIQUE_CARRIER',
    #                      'DEP_TIME_BLK',
    #                      'ARR_TIME_BLK']
     
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
    
        # Libération de mémoire
        #del data
        #del datanum
        #del data_x
        #del data_y
        
#        print("\nLinear")
#        a, b, c = linear(data_xtrain, data_xtest, data_ytrain, data_ytest)
#        
#        RMSE.append(round(a, 3))
#        #MSE.append(round(b, 3))
#        R2.append(round(c, 3))
#        
#        print("\nSGD")
#        a, b, c = sgd(data_xtrain, data_xtest, data_ytrain, data_ytest)
#        
#        RMSE.append(round(a, 3))
#        #MSE.append(round(b, 3))
#        R2.append(round(c, 3))
#        
#        print('\nLasso')
#        a, b, c = lasso(data_xtrain, data_xtest, data_ytrain, data_ytest)
#
#        RMSE.append(round(a, 3))
#        #MSE.append(round(b, 3))
#        R2.append(round(c, 3))
#    
#        print('\nElastic')
#        a, b, c = elastic(data_xtrain, data_xtest, data_ytrain, data_ytest)
#
#        RMSE.append(round(a, 3))
#        #MSE.append(round(b, 3))
#        R2.append(round(c, 3))
    


        log = all(data_xtrain, data_xtest, data_ytrain, data_ytest, compagnie, log)
        
        print(log)
    
def linear(data_xtrain, data_xtest, data_ytrain, data_ytest):

    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Train the model using the training sets
    regr.fit(data_xtrain, data_ytrain)
    
    # Make predictions using the testing set
    y_pred = regr.predict(data_xtest)
    
    RMSE = sqrt(mean_squared_error(data_ytest, y_pred))
    MSE = mean_squared_error(data_ytest, y_pred)
    R2 = r2_score(data_ytest, y_pred)
    
    # The mean squared error
    print("RMSE : %.3f" % RMSE)
    
    # The mean squared error
    print("Mean squared error: %.3f" % MSE)
    
    # Explained variance score: 1 is perfect prediction
    print('R2 score: %.3f' % R2)

    return RMSE, MSE, R2

def sgd(data_xtrain, data_xtest, data_ytrain, data_ytest):
    
    sgd = linear_model.SGDRegressor()
    
    sgd.fit(data_xtrain, data_ytrain)
    
    y_pred = sgd.predict(data_xtest)

    RMSE = sqrt(mean_squared_error(data_ytest, y_pred))
    MSE = mean_squared_error(data_ytest, y_pred)
    R2 = r2_score(data_ytest, y_pred)
    
    # The mean squared error
    print("RMSE : %.3f" % RMSE)
    
    # The mean squared error
    print("Mean squared error: %.3f" % MSE)
    
    # Explained variance score: 1 is perfect prediction
    print('R2 score: %.3f' % R2)

    return RMSE, MSE, R2

def svr(data_xtrain, data_xtest, data_ytrain, data_ytest):

    clf = SVR(kernel='linear', C=0.5, epsilon=0.2)
    
    clf.fit(data_xtrain, data_ytrain)
    
    y_pred = clf.predict(data_xtest)
    
    RMSE = sqrt(mean_squared_error(data_ytest, y_pred))
    MSE = mean_squared_error(data_ytest, y_pred)
    R2 = r2_score(data_ytest, y_pred)
    
    # The mean squared error
    print("RMSE : %.3f" % RMSE)
    
    # The mean squared error
    print("Mean squared error: %.3f" % MSE)
    
    # Explained variance score: 1 is perfect prediction
    print('R2 score: %.3f' % R2)

    return RMSE, MSE, R2

def poly():
    
    poly = PolynomialFeatures(degree=2)
    X_ = poly.fit_transform(X)
    predict_ = poly.fit_transform(predict)

def elastic(data_xtrain, data_xtest, data_ytrain, data_ytest):
    
    # import packages
    from sklearn.linear_model import ElasticNetCV
    from sklearn.metrics import mean_squared_error
    
    # specify the lasso regression model
    #print("Début du Fit")
    model=ElasticNetCV(cv=2, precompute=True).fit(data_xtrain, data_ytrain)
    #print("Fin du Fit")

    test_error = mean_squared_error(data_ytest, model.predict(data_xtest))

    rsquared_test=model.score(data_xtest,data_ytest)
        
    RMSE = sqrt(test_error)
    MSE = test_error
    R2 = rsquared_test
    
    # The mean squared error
    print("RMSE : %.3f" % RMSE)
    
    # The mean squared error
    print("Mean squared error: %.3f" % MSE)
    
    # Explained variance score: 1 is perfect prediction
    print('R2 score: %.3f' % R2)

    return RMSE, MSE, R2

def lasso(data_xtrain, data_xtest, data_ytrain, data_ytest):

    # import packages
    from sklearn.linear_model import LassoLarsCV
    from sklearn.metrics import mean_squared_error
    
    # specify the lasso regression model
    #print("Début du Fit")
    model=LassoLarsCV(cv=2, precompute=True).fit(data_xtrain, data_ytrain)
    #print("Fin du Fit")

    # MSE from training and test data
    train_error = mean_squared_error(data_ytrain, model.predict(data_xtrain))
    test_error = mean_squared_error(data_ytest, model.predict(data_xtest))
    
#    print ('training data MSE')
#    print(train_error)
#    print ('test data MSE')
#    print(test_error)
    
    # R-square from training and test data
    
    rsquared_train=model.score(data_xtrain,data_ytrain)
    rsquared_test=model.score(data_xtest,data_ytest)
    
#    print ('training data R-square')
#    print(rsquared_train)
#    
#    print ('test data R-square')
#    print(rsquared_test)
    
    RMSE = sqrt(test_error)
    MSE = test_error
    R2 = rsquared_test
    
    # The mean squared error
    print("RMSE : %.3f" % RMSE)
    
    # The mean squared error
    print("Mean squared error: %.3f" % MSE)
    
    # Explained variance score: 1 is perfect prediction
    print('R2 score: %.3f' % R2)

    return RMSE, MSE, R2

def testCV(data_xtrain, data_xtest, data_ytrain, data_ytest):
        
    model = linear_model.SGDRegressor()
    #axe_X = np.arange(0.1, 0.5, 0.1)
    #param_grid = {'n_neighbors':axe_X }
    
    param_grid = [{'alpha' : 10.0**-np.arange(1,7),
                   'l1_ratio':[.05, .15, .5, .7, .9, .95, .99, 1]}]
        
    #dictionnaire = {'C':axe_X, 'epsilon':axe_X }

    score = 'neg_mean_squared_error'

    clf = GridSearchCV(model, param_grid=param_grid, cv=5, scoring=score, refit=True)
    
    clf.fit(data_xtrain, data_ytrain)
    best_params = clf.best_params_
    score = sqrt(abs(clf.best_score_))
    
    score_MSE = []
    
    for a, c in zip(clf.cv_results_['mean_test_score'], clf.cv_results_['params']):
        print("\t%s %s %s pour %s" % ('\tGRIDSCORES\t',  "RMSE" , sqrt(abs(a)), c))
        score_MSE.append(sqrt(abs(a)))
    
    print('\n\t\t%s\t%f' % (str(best_params), abs(score)))
    
#    plt.figure(1, figsize=(15, 9))
#
#    plt.plot(10.0**-np.arange(1,7), score_MSE, label="MSE", color='g')
#    
#    plt.title("Grid Search Scores", fontsize=20, fontweight='bold')
#    plt.xlabel('RMSE', fontsize=16)
#    plt.ylabel('CV Average Score', fontsize=16)
#    plt.legend(loc="best", fontsize=15)
#    plt.grid('on')
#    plt.show()
    
def all(data_xtrain, data_xtest, data_ytrain, data_ytest, compagnie, log):
        
    from sklearn.externals import joblib

    log_cols=["Classifier", "Id", "RMSE", "R2"]
    score = 'neg_mean_squared_error'
        
    classifiers = [SGDRegressor(),
                   LinearRegression(),
                   ElasticNetCV(),
                   LassoCV(),
                   OrthogonalMatchingPursuitCV(),
                   RidgeCV(scoring=score)]
    
    for clf in classifiers:
        #clf.fit(data_xtrain, data_ytrain)
        name = clf.__class__.__name__
        
        fichier = _DOSSIERTRAVAIL + "\\" + name + "_" + compagnie + ".pkl"
        
        #joblib.dump(clf, fichier)
        
        clf = joblib.load(fichier) 

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
