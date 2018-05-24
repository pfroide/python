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

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.moses import MosesDetokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from matplotlib import pyplot as plt, cm as cm
from yellowbrick.text import FreqDistVisualizer
#from wordcloud import WordCloud

# Lieu où se trouve le fichier
_DOSSIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\p6\\'
_DOSSIERTRAVAIL = 'C:\\Users\\Toni\\python\\python\\Projet_6\\images'

def creer_tfidfvectorizer(text):
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

def creer_countvectorizer(text):
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

def display_topics(model, feature_names, no_top_words):
    """
    TBD
    """

    for topic_idx, topic in enumerate(model.components_):
        print("Topic", topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

def reduction_dimension(data):
    """
    TBD
    """

    from sklearn.decomposition import TruncatedSVD

    for i in [500, 1500, 2500, 3000]:
        svd = TruncatedSVD(n_components=i,
                           n_iter=7,
                           random_state=42)

        svd.fit(data)

        #print(svd.explained_variance_ratio_)
        print(i)
        print(round(100*svd.explained_variance_ratio_.sum(), 2))
        #print(svd.singular_values_)

#def generate_wordcloud(data):
#    """
#    Simple WordCloud
#    """
#
#    wordcloud = WordCloud(background_color='black',
#                          #stopwords=stopwords,
#                          max_words=20,
#                          max_font_size=20,
#                          relative_scaling=1,
#                          min_font_size=1,
#                          scale=3,
#                          random_state=1 # chosen at random by flipping a coin; it was heads
#                         ).generate(str(data))
#
#    fig = plt.figure(1, figsize=(12, 12))
#    plt.axis('off')
#
#    #if title:
#    #    fig.suptitle(title, fontsize=20)
#    #    fig.subplots_adjust(top=2.3)
#
#    plt.imshow(wordcloud)
#    plt.show()

def comptage(data):
    """
    TBD
    """
    import collections

    count = collections.Counter()

    stop_words = set(stopwords.words('english'))

    for sentence in data:
        sentence = word_tokenize(sentence.lower())
        for word in sentence:
            if (not word in stop_words) and word.isalpha():
                count[word] += 1

    return count

def main():
    """
    Fonction principale
    """

    # Récupération du dataset
    fichier = 'QueryResults.csv'
    data = pd.read_csv(_DOSSIER + fichier, error_bad_lines=False)

    #
    data = data[0:10000]

    # Nouveau dataframe qui prendra les résultats en entrée
    new_df = pd.DataFrame()

    # Données manquantes
    donnees_manquantes(data, "missing_data_1")

    # Suppression des balises html et des parties de code
    data['Body'] = [suppr_html_code(x) for x in data['Body']]

    # Comptage du nombre d'occurence
    cpt = comptage(data['Body'])

    # Liste des stop words anglais
    least_used = set([word for word in cpt if cpt[word] < 5])

    stop_words = set(stopwords.words('english')) \
                | set([word for word, freq, in cpt.most_common(100)]) \
                | least_used

    # Suppression des pluriels et des stop words
    # Rétrecissment du dataset
    new_df['Sentences'] = [fct_nltk(x, stop_words) for x in data['Body']]

    # On est obligé de detokenizer pour créer la matrice finale
    detokenizer = MosesDetokenizer()
    new_df['Sentences'] = [detokenizer.detokenize(x, return_str=True) for x in new_df['Sentences']]

    # Création de la matrice finale
    matrixnum_count, names_count = creer_countvectorizer(new_df['Sentences'])
    # Conversion de la matrice finale en dataframe pour facilité d'usage
    matrixnum_count = pd.DataFrame(matrixnum_count, columns=names_count)

    # Run LDA
    lda = LatentDirichletAllocation(n_components=20,
                                    max_iter=5,
                                    learning_method='online',
                                    learning_offset=50,
                                    random_state=0)

    #
    lda.fit(matrixnum_count)

    #
    display_topics(lda, names_count, 10)

    #
    #generate_wordcloud(matrixnum_count)
    #| set([word for word, freq, in count.most_common(100)])

    # Test de visualisation
    visualizer = FreqDistVisualizer(features=names_count,
                                    n=25,
                                    orient='v',
                                    color='g')
    visualizer.fit(matrixnum_count)
    visualizer.poof()

    # Création de la matrice finale
    matrixnum_tfidf, names_tfidf = creer_tfidfvectorizer(new_df['Sentences'])
    # Conversion de la matrice finale en dataframe pour facilité d'usage
    matrixnum_tfidf = pd.DataFrame(matrixnum_tfidf, columns=names_tfidf)

    # Run NMF
    nmf = NMF(n_components=20,
              random_state=1,
              alpha=.1,
              l1_ratio=.5,
              init='nndsvd')

    #
    nmf.fit(matrixnum_tfidf)

    #
    display_topics(nmf, names_tfidf, 10)

    # Test de visualisation
    visualizer = FreqDistVisualizer(features=names_tfidf,
                                    n=25,
                                    orient='v',
                                    color='g')
    visualizer.fit(matrixnum_tfidf)
    visualizer.poof()
