# -*- coding: utf-8 -*-
"""
Created on Mon May 21 18:49:06 2018

@author: Toni
"""

# On importe les librairies dont on aura besoin pour ce tp
import nltk
import os
import pandas as pd

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from mosestokenizer import MosesDetokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Lieu où se trouve le fichier
_DOSSIER = '/home/toni/Bureau/cnn/stories/'
_DOSSIERTRAVAIL = '/home/toni/python/Exercices/VI. Texte/'

def creer_countvectorizer(text):
    """
    Fonction de création de la matrice numérique
    Comptage de fréquence
    """

    # Création de l'objet
    vectorizer = CountVectorizer()

    # Fit du texte d'entrée, et mis au format tableau
    liste_mots = vectorizer.fit_transform(text).toarray()

    # On ressort le tableau, et la liste des mots
    return liste_mots, vectorizer.get_feature_names()

def creer_tfidfvectorizer(text):
    """
    Fonction de création de la matrice
    Comptage global d'occurence
    """

    # Création de l'objet
    t_vectorizer = TfidfVectorizer()

    # Fit du texte d'entrée, et mis au format tableau
    liste_mots = t_vectorizer.fit_transform(text).toarray()

    # On ressort le tableau, et la liste des mots
    return liste_mots, t_vectorizer.get_feature_names()

    #print(X.toarray())
    #print(Y.toarray())

    #print(vectorizer.get_feature_names())
    #print(t_vec.get_feature_names())

    #dense = Y.todense()
    #denselist = dense.tolist()
    #df = pd.DataFrame(denselist)
    #s = pd.Series(df.loc[0])
    #s[s > 0].sort_values(ascending=False)

def fct_nltk(text, stop_words):
    """
    Fonction pour supprimer :
        les step words
        la ponctuation
        les majuscules
        les pluriels
    """

    # Création de l'objet
    lemma = nltk.wordnet.WordNetLemmatizer()

    # Tokenization et mise en minuscule
    words = word_tokenize(text.lower())

    # Suppression des pluriels et de la ponctuation. Boule pour toutes les lignes
    new_sentence = [lemma.lemmatize(x) for x in words if (not x in stop_words) and x.isalpha()]

    # Sortie
    return new_sentence

def suppr_html_code(fichier):
    """
    Fonction de suppression des balises HTML et des parties de code
    """

    # Suppression des balises HTML
    soupe = BeautifulSoup(fichier, "lxml")

    # Recherche des balises de 'code'
    liste = soupe.findAll('code')

    # Suppression des données qui sont entre les balises de code
    for balise in liste:
        balise.decompose()

    # Sortie formatée en texte
    return soupe.text

def main():
    """
    Fonction principale
    """

    # Récupération du dataset
    liste = os.listdir(_DOSSIER)
    data = pd.DataFrame()

    for doc in liste:
        datatemp = pd.read_table(_DOSSIER + doc,
                                        header=None,
                                        error_bad_lines=False,
                                        engine='python')

        data = pd.concat([data, datatemp])

    data = data.rename(columns={0: "Body"}).reset_index(drop=True)

    # Nouveau dataframe qui prendra les résultats en entrée
    new_df = pd.DataFrame()

    # Suppression des balises html et des parties de code
    data['Body'] = [suppr_html_code(x) for x in data['Body']]

    # Suppression des pluriels et des stop words
    stop_words = set(stopwords.words('english'))
    new_df['Sentences'] = [fct_nltk(x, stop_words) for x in data['Body']]

                # On est obligé de detokenizer pour créer la matrice finale
    detokenizer = MosesDetokenizer()
    new_df['Sentences'] = [detokenizer(x) for x in new_df['Sentences']]

    # Création de la matrice finale
    matrixnum_count, names_count = creer_countvectorizer(new_df['Sentences'])
    # Conversion de la matrice finale en dataframe pour facilité d'usage
    matrixnum_count = pd.DataFrame(matrixnum_count, columns=names_count)

    # Création de la matrice finale
    matrixnum_tfidf, names_tfidf = creer_tfidfvectorizer(new_df['Sentences'])
    # Conversion de la matrice finale en dataframe pour facilité d'usage
    matrixnum_tfidf = pd.DataFrame(matrixnum_tfidf, columns=names_tfidf)

    # Exportation
    data.to_csv(_DOSSIER + 'export.csv')
