# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:51:35 2018

@author: Toni
"""

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import neighbors, metrics
from sklearn.neighbors import NearestNeighbors

_CHEMIN = 'C:\\Users\\Toni\\python\\python\\Exercices\\IV. Vins\\'
_ROUGE = 'winequality-red.csv'
_BLANC = 'winequality-white.csv'

param_neigh = range(3,16,2)

nb_fold = 5

# Récuparation du dataset
data = pd.read_csv(_CHEMIN + _BLANC, sep=";")

def ma_fonction(param_neigh):
    
    # Mise en forme du dataset
    X = data.as_matrix(data.columns[:-1])
    y = data.as_matrix([data.columns[-1]])
    y = y.flatten()
    y_class = np.where(y < 6, 0, 1)

    # 30% des données dans le jeu de test
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_class, test_size=0.3)

    std_scale = preprocessing.StandardScaler().fit(X_train)
    X_train_std = std_scale.transform(X_train)
    X_test_std = std_scale.transform(X_test)
    
    for p in param_neigh:
        nbrs = NearestNeighbors(n_neighbors=p).fit(X_train_std)
        distances, indices = nbrs.kneighbors(X_train_std)
        print(distances)
        print(indices)


def fonction_valid_croise(param_grid):
    """
    TBD
    """

    # Choisir un score à optimiser, ici l'accuracy (proportion de prédictions correctes)
    score = 'accuracy'

    # Mise en forme du dataset
    X = data.as_matrix(data.columns[:-1])
    y = data.as_matrix([data.columns[-1]])
    y = y.flatten()
    y_class = np.where(y < 6, 0, 1)

    # 30% des données dans le jeu de test
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_class, test_size=0.3)

    std_scale = preprocessing.StandardScaler().fit(X_train)
    X_train_std = std_scale.transform(X_train)
    X_test_std = std_scale.transform(X_test)

    # Créer un classifieur kNN avec recherche d'hyperparamètre par validation croisée
    clf = model_selection.GridSearchCV(neighbors.KNeighborsClassifier(), # un classifieur kNN
                                       param_grid, # hyperparamètres à tester
                                       cv=5,
                                       scoring=score # score à optimiser
                                      )

    # cv=5, # nombre de folds de validation croisée
    # Optimiser ce classifieur sur le jeu d'entraînement
    clf.fit(X_train_std, y_train)

    # Afficher le(s) hyperparamètre(s) optimaux
    print("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement : ", clf.best_params_)

    # Afficher les performances correspondantes
    print("Résultats de la validation croisée :")

    for mean, std, params in zip(clf.cv_results_['mean_test_score'], # score moyen
                                 clf.cv_results_['std_test_score'], # écart-type du score
                                 clf.cv_results_['params'] # valeur de l'hyperparamètre
                                ):

        print("\t%s = %0.3f (+/-%0.03f) for %r" % (score, # critère utilisé
                                                   mean, # score moyen
                                                   std * 2, # barre d'erreur
                                                   params # hyperparamètre
                                                  ))

    y_pred = clf.predict(X_test_std)

    print("\nSur le jeu de test : %0.3f" % metrics.accuracy_score(y_test, y_pred))

def main():
    
    # Fixer les valeurs des hyperparamètres à tester
    param_grid = {'n_neighbors':[3, 5, 7, 9, 11, 13, 15]}

    fonction_valid_croise(param_grid)