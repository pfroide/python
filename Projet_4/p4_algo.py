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
        
    #resultsRMSE = pd.DataFrame({'RMSE' : []})
    #resultsMSE = pd.DataFrame({'MSE' : []})
    #resultsR2 = pd.DataFrame({'R2' : []})
    
    RMSE = []
    #MSE = []
    R2 = []

    liste = datanum['UNIQUE_CARRIER'].unique()
    
    for compagnie in liste:
        datanum = data.copy()

        print('Pour la compagnie', compagnie)
        
        datanum = datanum[datanum['UNIQUE_CARRIER']==compagnie]
        del datanum['UNIQUE_CARRIER']

    # Données manquantes
#    missing_data = datanum.isnull().sum(axis=0).reset_index()
#    missing_data.columns = ['column_name', 'missing_count']
#    missing_data['filling_factor'] = (datanum.shape[0]-missing_data['missing_count'])/datanum.shape[0]*100
#    missing_data.sort_values('filling_factor').reset_index(drop=True)

#    liste_a_supprimer = ['AIRLINE_ID', 'AIR_TIME', 'ARR_TIME', 'CARRIER_DELAY']
#    liste_a_supprimer.extend(['DAY_OF_MONTH', 'DEP_TIME', 'DISTANCE', 'LATE_AIRCRAFT_DELAY'])
#    liste_a_supprimer.extend(['NAS_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY'])
#    liste_a_supprimer.extend(['ORIGIN_CITY_NAME', 'ORIGIN_AIRPORT_ID'])
#    liste_a_supprimer.extend(['FL_DATE', 'DEST_CITY_NAME', 'DEST_AIRPORT_ID', 'DEP_DELAY'])

#    for donnee in liste_a_supprimer:
#        del datanum[donnee]

        # Transposition en 0 et 1 des valeurs non-numériques
        liste_criteres = ['DEST', 
                          'ORIGIN',
                          'DEP_TIME_BLK',
                          'ARR_TIME_BLK',
                          'DISTANCE_GROUP',
                          'DAY_OF_WEEK', 
                          'MONTH']

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
        
        print("\nLinear")
        a, b, c = linear(data_xtrain, data_xtest, data_ytrain, data_ytest)
        
        RMSE.append(round(a, 3))
        #MSE.append(round(b, 3))
        R2.append(round(c, 3))
        
        print("\nSGD")
        a, b, c = sgd(data_xtrain, data_xtest, data_ytrain, data_ytest)
        
        RMSE.append(round(a, 3))
        #MSE.append(round(b, 3))
        R2.append(round(c, 3))
        
        print('\nLasso')
        a, b, c = lasso(data_xtrain, data_xtest, data_ytrain, data_ytest)

        RMSE.append(round(a, 3))
        #MSE.append(round(b, 3))
        R2.append(round(c, 3))
    
    print(sum(RMSE)/len(RMSE))
    print(sum(R2)/len(R2))
    
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

def lasso(data_xtrain, data_xtest, data_ytrain, data_ytest):

    # import packages
    from sklearn.linear_model import LassoLarsCV
    from sklearn.metrics import mean_squared_error
    
    # specify the lasso regression model
    print("Début du Fit")
    model=LassoLarsCV(cv=10, precompute=True).fit(data_xtrain, data_ytrain)
    print("Fin du Fit")
    
#    # print variable names and regression coefficients
#    #print(dict(zip(data_xtrain.columns, model.coef_)))
#    
#    #plot mean square error for each fold
#    m_log_alphascv = -np.log10(model.cv_alphas_)
#    plt.figure()
#    plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
#    plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
#    plt.axvline(-np.log10(model.alpha_), color='k', label='alpha CV')
#    
#    plt.legend()
#    plt.xlabel('-log(alpha)')
#    plt.ylabel('Mean squared error')
#    plt.title('Mean squared error on each fold')
#    plt.show()
#
#    # In scikit learn lambda penalty is set by alpha_ attribute of model we select cv=10 in LassoLarsCV() function that is no  of K fold Validation. This graph shows the mean squared error of every fold with dotted line and average mean squared error of k fold is shown in solid line. The Vertical dotted black line choose optimum point having lowest bias and variance and this is used as  lambda penalty for shrinking coefficient.-np.log10()  transformation is applied to model.alpha_ to make it easily understandable
#    
#    # plot coefficient progression
#    m_log_alphas = -np.log10(model.alphas_)
#    ax = plt.gca()
#    plt.plot(m_log_alphas, model.coef_path_.T)
#    plt.axvline(-np.log10(model.alpha_), color='k', label='alpha CV')
#    plt.ylabel('Regression Coefficients')
#    plt.legend()
#    plt.xlabel('-log(alpha)')
#    plt.title('Regression Coefficients Progression for Lasso Paths')
#    plt.show()
    
    # This graphs shows that relative importance of every variable and how their coefficient changes when new variable enters into the model. Sgpt Alamine Aminotransferase enter into the model first because it is more negatively correlated. The lambda penalty displayed between 2.9 to 3 in dotted vertical line controls the coefficient of every non zero coefficient attribute
    
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