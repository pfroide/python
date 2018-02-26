# -*- coding: utf-8 -*-
"""
Created on Sat Fev 24 19:51:35 2018

@author: Toni

Comme dit plus haut votre objectif sera de déterminer quelle est l’espèce de l’arbre
 à laquelle appartient la feuille.

Les caractéristiques extraites des images des feuilles sont essentiellement
 3 vecteurs de dimension 64 (margin, shape & texture).

Utilisez bien l’ensemble des notions vues dans cette
 section (choix des hyperparamètres, régularisation) afin de pouvoir 
 obtenir les meilleurs performances de classification possible.

Vous devrez donc :

* Créer une baseline de performances avec le K-NN
* Utiliser le SVM multiclasse avec différents paramètres et l’optimiser
* Une critique et visualisation des performances des modèles sur ce jeu de données
* Une sélection d’un modèle final à partir des performances

# id - an anonymous id unique to an image
# margin - each of the 64 attribute vectors for the margin feature
# shape - each of the 64 attribute vectors for the shape feature
# texture - each of the 64 attribute vectors for the texture feature

"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import preprocessing
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit

# Sklearn Classifier Showdown
# Simply looping through 10 out-of-the box classifiers and printing the results. Obviously, these will perform much better after tuning their hyperparameters, but this gives you a decent ballpark idea.

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

_CHEMIN = 'C:\\Users\\Toni\\python\\python\\Exercices\\Feuilles\\'
_FEUILLE = 'Dataset_feuilles_1.csv'

# Stratified Train/Test Split
# Stratification is necessary for this dataset because there is a relatively large number of classes (100 classes for 990 samples). This will ensure we have all classes represented in both the train and test indices.
def premiere_partie():

    # Récuparation du dataset
    data = pd.read_csv(_CHEMIN + _FEUILLE, sep=",")

    # On récupère les features d'un côté...
    X = data.copy()
    X = X.drop(['species', 'id'], axis=1)  

    # et les labels de l'autre
    y = data.iloc[:,1] 
    le = LabelEncoder().fit(y) 
    y = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    classifiers = [
        KNeighborsClassifier(4),
        LinearSVC(),
        SVC(C=10000),
        #NuSVC(probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]
    
    # Logging for Visual Comparison
    log_cols=["Classifier", "Accuracy"]
    log = pd.DataFrame(columns=log_cols)
    
    for clf in classifiers:
        clf.fit(X_train, y_train)
        name = clf.__class__.__name__
        
        print("="*30)
        print(name, '\n')

        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        print("Accuracy: {:.2%}".format(acc))

        log_entry = pd.DataFrame([[name, acc*100]], columns=log_cols)
        log = log.append(log_entry)
        
    print("="*30)
    
    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
    
    plt.xlabel('Accuracy %')
    plt.title('Classifier Accuracy')
    plt.show()
 
def main():
    
    premiere_partie()
