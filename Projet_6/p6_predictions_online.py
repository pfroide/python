#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 18:23:44 2018

@author: toni
"""

# On importe les librairies dont on aura besoin pour ce tp
import json
import pandas as pd
from flask import Flask, request
from sklearn.externals import joblib
#from mosestokenizer import MosesDetokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import wordnet
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings("ignore")

# Lieu où se trouve le fichier
_DOSSIERWEB = '/home/toni/python/Projet_6/images/'

# Création de la route d'entrée de l'API

def fct_nltk(text, stop_words):
    """
    Fonction pour supprimer :
        les step words
        la ponctuation
        les majuscules
        les pluriels
    """

    # Création de l'objet
    lemma = wordnet.WordNetLemmatizer()

    # Tokenization et mise en minuscule
    words = word_tokenize(text.lower())

    # Suppression des pluriels et de la ponctuation. Boule pour toutes les lignes
    new_sentence = [lemma.lemmatize(x) for x in words if (not x in stop_words) and x.isalpha()]

    # Sortie
    return new_sentence

def predict(texte):

    # Création du dataframe vide pour avoir le nom des colonnes
    fichier = "RandomForestClassifier_30_10.pkl"

    # On va chercher le dump
    classif = joblib.load(_DOSSIERWEB + fichier)

    # Nettoyage du texte
    fichier2 = _DOSSIERWEB + "stop.pkl"
    stop_words = joblib.load(fichier2)
    texte = fct_nltk(texte, stop_words)
    texte = " ".join(texte)

    # Création de l'objet
    t_vectorizer = TfidfVectorizer(min_df=0.01)

    # Fit du texte d'entrée, et mis au format tableau
    matrix = t_vectorizer.fit_transform([texte]).toarray()
    liste_tags = t_vectorizer.get_feature_names()
    matrix = pd.DataFrame(matrix, columns=liste_tags)

    #
    fichier3 = _DOSSIERWEB + "index.pkl"
    index_c = joblib.load(fichier3)

    for i in index_c:
        if i not in liste_tags:
            matrix[i]=0

    for i in matrix.columns:
        if i not in index_c:
            del matrix[i]

    fichier4 = _DOSSIERWEB + "colonnes.pkl"
    colonnes = joblib.load(fichier4)

    # Predictions
    predictions = classif.predict_proba(matrix)
    predictions = pd.DataFrame(predictions, columns=colonnes)

    df_prevision_finale = pd.DataFrame()

    for p in predictions.index:
        temp = predictions.loc[p]
        temp = temp[temp>0]
        temp = temp.nlargest(5)
        temp = temp.reset_index()
        temp = temp.rename(columns={"index": 'autre'})

        # On supprime ceux qui ne sont pas dans le body de la question
        #for k, v in enumerate(temp['index']):
        #    mot = ' ' + v + ' '
        #    if mot not in texte:
        #        temp = temp.drop(k)

        df_prevision_finale = df_prevision_finale.append(temp['autre'])

    df_prevision_finale = pd.DataFrame(df_prevision_finale).reset_index(drop=True)

    # On prends les lignes une par une
    ligne = df_prevision_finale.copy()

    # Variable list qui va prendre les résultats des prédictions
    predicted_label = []

    for word in ligne.loc[0]:
    #    if not pd.isna(word):
        predicted_label.append(word)

    print("Predicted label :", predicted_label, "\n")

    # Résultat de la prédiction, transformation en json
    dico = {}
    for nb, key in enumerate(predicted_label):
        dico[str(nb)] = str(key)

    # Transformation en format JSON
    json_dico = json.dumps(dico, indent=4, separators=(',', ': '))

    # On renvoie la valeur trouvée
    return json_dico
