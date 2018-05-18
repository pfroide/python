# -*- coding: utf-8 -*-
"""
Created on Thu May 17 21:51:03 2018

@author: Toni
"""

# On importe les librairies dont on aura besoin pour ce tp
import nltk
import numpy as np
import pandas as pd
#nltk.download()

from nltk.stem.porter import *
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt, cm as cm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Lieu où se trouve le fichier
_DOSSIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\p6\\'
_DOSSIERTRAVAIL = 'C:\\Users\\Toni\\python\\python\\Projet_6\\images'

def creer_matrix(text):
    """
    TBD
    """

    vectorizer = CountVectorizer()
    t_vec = TfidfVectorizer()

    X = vectorizer.fit_transform(text)
    Y = t_vec.fit_transform(text)

    #print(X.toarray())
    #print(Y.toarray())

    #print(vectorizer.get_feature_names())
    #print(t_vec.get_feature_names())

    return X.toarray()

    #dense = Y.todense()
    #denselist = dense.tolist()
    #df = pd.DataFrame(denselist)
    #s = pd.Series(df.loc[0])
    #s[s > 0].sort_values(ascending=False)

def fct_nltk(text):
    """
    TBD
    """

    lemma = nltk.wordnet.WordNetLemmatizer()

    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())

    new_sentence = [lemma.lemmatize(x) for x in words if ((not x in stop_words) and x.isalpha())]

    return new_sentence

def suppr_html_code(fichier):
    """
    TBD
    """

    soupe = BeautifulSoup(fichier, "lxml")

    liste = soupe.findAll('code')

    for balise in liste:
        balise.decompose()

    return soupe.text

def donnees_manquantes(data, nom):
    """
    Données manquantes
    """

    # Données manquantes
    fichier_save = _DOSSIERTRAVAIL + '\\' + nom + '.csv'
    missing_data = data.isnull().sum(axis=0).reset_index()
    missing_data.columns = ['column_name', 'missing_count']
    missing_data['filling_factor'] = (data.shape[0]-missing_data['missing_count'])/data.shape[0]*100
    print(missing_data.sort_values('filling_factor').reset_index(drop=True))
    missing_data.sort_values('filling_factor').reset_index(drop=True).to_csv(fichier_save)

    # Transposition du dataframe de données pour l'analyse univariée
    fichier_save = _DOSSIERTRAVAIL + '\\' + 'transposition.csv'
    data_transpose = data.describe().reset_index().transpose()
    print(data_transpose)
    data_transpose.to_csv(fichier_save)

def main():
    """
    Fonction principale
    """

    # Récupération du dataset
    fichier = 'QueryResults.csv'
    data = pd.read_csv(_DOSSIER + fichier, error_bad_lines=False)

    #
    data = data[0:500]

    #
    new_df = pd.DataFrame()
    matrix_num = pd.DataFrame()

    # Données manquantes
    donnees_manquantes(data, "missing_data_1")

    #
    data['Body'] = [suppr_html_code(x) for x in data['Body']]
    new_df['Sentences'] = [fct_nltk(x) for x in data['Body']]
    matrix_num['numbers'] = [creer_matrix(x) for x in new_df['Sentences']]
