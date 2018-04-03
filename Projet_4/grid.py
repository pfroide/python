# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 19:39:06 2018

@author: Toni
"""

import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn import datasets
    
class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
    
    def fit(self, X, y, cv=3, n_jobs=1, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs, 
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X,y)
            self.grid_searches[key] = gs    
    
    def score_summary(self, sort_by='max_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),  
            }
            #return pd.Series(dict(params.items() + d.items()))
            return pd.Series(dict(params.items() | d.items()))    
                      
        rows = [row(k, gsc.cv_validation_scores, gsc.parameters) 
                     for k in self.keys
                     for gsc in self.grid_searches[k].grid_scores_]
        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)
        
        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        
        return df[columns]

def main(): 

    diabetes = datasets.load_diabetes()
    X_diabetes = diabetes.data
    y_diabetes = diabetes.target
    
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    
    models2 = { 
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso()
    }
    
    params2 = { 
        'LinearRegression': { },
        'Ridge': { 'alpha': 10.0**-np.arange(1,7) },
        'Lasso': { 'alpha': 10.0**-np.arange(1,7) }
    }
    
    helper2 = EstimatorSelectionHelper(models2, params2)
    helper2.fit(X_diabetes, y_diabetes, n_jobs=-1)
    
    helper2.score_summary()

