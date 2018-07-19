# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

# On importe les librairies dont on aura besoin pour ce tp
import warnings
import sklearn

from sklearn.datasets import fetch_rcv1
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

# Pour ne pas avoir les warnings lors de la compilation
warnings.filterwarnings("ignore")

def main():
    """
    fonction principale
    """

    # Séparation des données train/test
    print("PREPARING DATA ...")
    train = fetch_rcv1(subset='train')
    test = fetch_rcv1(subset='test')

    # Récupération des parties qui nous intéressent
    X_train = train.data
    X_test = test.data
    y_train = train.target
    y_target = test.target

    # C'est trop de données, donc je réduis un peu.
    # A mettre en commentaire si nécessaire
    X_train = X_train[:10000]
    X_test = X_test[:10000]
    y_train = y_train[:10000]
    y_target = y_target[:10000]

    print("... done\n")

    # 1ère évaluateur
    print("CREATING LINEAR_SVC")
    clf = LinearSVC(random_state=0)
    y_test = OneVsRestClassifier(clf).fit(X_train, y_train).predict(X_test)

    # Création des données pour l'évaluation
    test_eval = y_test.toarray()
    target_eval = y_target.toarray()

    # Fonction qui affiche les scores
    score(target_eval, test_eval)
    print("... done\n")

    # 2nd évaluateur
    print("CREATING KNeighborsClassifier")
    clf = KNeighborsClassifier(3)
    y_test = OneVsRestClassifier(clf).fit(X_train, y_train).predict(X_test)

    # Création des données pour l'évaluation
    test_eval = y_test.toarray()
    target_eval = y_target.toarray()

    # Fonction qui affiche les scores
    score(target_eval, test_eval)
    print("... done\n")

    # 3ème évaluateur
    print("CREATING SGDClassifier")
    clf = SGDClassifier()
    y_test = OneVsRestClassifier(clf).fit(X_train, y_train).predict(X_test)

    # Création des données pour l'évaluation
    test_eval = y_test.toarray()
    target_eval = y_target.toarray()

    # Fonction qui affiche les scores
    score(target_eval, test_eval)
    print("... done\n")

def score(target_eval, test_eval):
    """
    Fonction qui affiche tous les scores
    """

#    amacro = sklearn.metrics.average_precision_score(target_eval, test_eval, average='macro')
#    print("Average macro: " + str(amacro))

#    amicro = sklearn.metrics.average_precision_score(target_eval, test_eval, average='micro')
#    print("Average micro: " + str(amicro))

#    micro = sklearn.metrics.precision_score(test_eval, target_eval, average='micro')
#    print("micro-precision: " + str(micro))

#    macro = sklearn.metrics.precision_score(test_eval, target_eval, average='macro')
#    print("macro-precision: " + str(macro))

    recall = sklearn.metrics.recall_score(test_eval, target_eval, average='micro')
    print("recall: " + str(round(recall, 3)))

    f1 = sklearn.metrics.f1_score(test_eval, target_eval, average='micro')
    print("f1: " + str(round(f1, 3)))
