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

_CHEMIN = 'C:\\Users\\Toni\\python\\python\\Exercices\\IV. Vins\\'
_ROUGE = 'winequality-red.csv'
_BLANC = 'winequality-white.csv'

# Récuparation du dataset
data = pd.read_csv(_CHEMIN + _BLANC, sep=";")

def ma_fonction(p, nb_fold):
    """
    TBD
    """

    # Partie de découpage
    taille_fold = int(len(data)/nb_fold)

    somme = 0

    for i in range(0, nb_fold):
        # On gére d'abord les deux particuliers
        if i == 0:
            data_test = data[:(i+1)*taille_fold]
            data_train = data[(i+1)*taille_fold:]

        elif i == (nb_fold-1):
            data_test = data[i*taille_fold:]
            data_train = data[:i*taille_fold]

        # Cas  général : On merge les deux bouts si on a extrait au départ un bout du milieu
        elif i != (nb_fold-1):
            data_test = data[i*taille_fold:(i+1)*taille_fold]
            data_train1 = data[:i*taille_fold]
            data_train2 = data[(i+1)*taille_fold:]

            frames = [data_train1, data_train2]
            data_train = pd.concat(frames)

        # Mise en forme du dataset
        x_train = data_train.as_matrix(data_train.columns[:-1])
        y_train = data_train.as_matrix([data_train.columns[-1]])

        # Mise en forme du dataset
        x_test = data_test.as_matrix(data_test.columns[:-1])
        y_test = data_test.as_matrix([data_test.columns[-1]])
        
        # Classification en binaire
        y_test = np.where(y_test < 6, 0, 1)

        # Scale
        std_scale = preprocessing.StandardScaler().fit(x_train)
        x_train_std = std_scale.transform(x_train)
        x_test_std = std_scale.transform(x_test)

        # Prédictions
        nbrs = neighbors.KNeighborsClassifier(n_neighbors=p).fit(x_train_std, y_train)
        y_pred = nbrs.predict(x_test_std)

        # Classification en binaire
        y_pred = np.where(y_pred < 6, 0, 1)

        # Cumul pour ensuite diviser et avoir la moyenne du score
        somme = somme + metrics.accuracy_score(y_test, y_pred)

    # Score
    return somme/nb_fold

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
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y_class, test_size=0.3)

    std_scale = preprocessing.StandardScaler().fit(x_train)
    x_train_std = std_scale.transform(x_train)
    x_test_std = std_scale.transform(x_test)

    # Créer un classifieur kNN avec recherche d'hyperparamètre par validation croisée
    clf = model_selection.GridSearchCV(neighbors.KNeighborsClassifier(), # un classifieur kNN
                                       param_grid, # hyperparamètres à tester
                                       cv=5,
                                       scoring=score # score à optimiser
                                      )

    # cv=5, # nombre de folds de validation croisée
    # Optimiser ce classifieur sur le jeu d'entraînement
    clf.fit(x_train_std, y_train)

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

    y_pred = clf.predict(x_test_std)

    print("\nSur le jeu de test : %0.3f" % metrics.accuracy_score(y_test, y_pred))

def main():
    """
    TBD
    """

    # Fixer les valeurs des hyperparamètres à tester
    param_grid = [3, 5, 7, 9, 11, 13, 15]

    liste_finale = dict()

    for p in param_grid:
        liste_finale[p] = (ma_fonction(p, 5))

    print(liste_finale.items())

    import operator
    indicemax = max(liste_finale.items(), key=operator.itemgetter(1))[0]

    print('Configuration optimale pour', indicemax, 'neighbors :', round(liste_finale[indicemax], 3))
