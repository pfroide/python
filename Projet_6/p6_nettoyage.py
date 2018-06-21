# -*- coding: utf-8 -*-
"""
Created on Thu May 17 21:51:03 2018

@author: Toni
"""

# On importe les librairies dont on aura besoin pour ce tp
import sys
import numpy as np
import collections
import pandas as pd

from bs4 import BeautifulSoup
from nltk import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.moses import MosesDetokenizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from yellowbrick.text import FreqDistVisualizer

# Lieu où se trouve le fichier
if sys.platform == "windows":
    _FICHIER = 'QueryResults.csv'
    _DOSSIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\p6\\'
    _DOSSIERTRAVAIL = 'C:\\Users\\Toni\\python\\python\\Projet_6\\images'

elif sys.platform == "linux":
    _FICHIER = 'stackoverflow_train_dataset.csv'
    _DOSSIER = '/home/toni/Bureau/'
    _DOSSIERTRAVAIL = '/home/toni/python/Projet_6/images/'

# faire avec et sans stemming dans les deux cas

def creer_countvectorizer(text):
    """
    Fonction de création de la matrice
    Comptage global d'occurence
    """

    # Création de l'objet
    vectorizer = CountVectorizer()

    # Fit du texte d'entrée, et mis au format tableau
    liste_mots = vectorizer.fit_transform(text).toarray()

    # On ressort le tableau, et la liste des mots
    return liste_mots, vectorizer.get_feature_names()

def creer_tfidfvectorizer(text):
    """
    Fonction de création de la matrice numérique
    Comptage de fréquence
    """

    # Création de l'objet
    t_vectorizer = TfidfVectorizer(min_df=0.01)

    # Fit du texte d'entrée, et mis au format tableau
    liste_mots = t_vectorizer.fit_transform(text).toarray()

    # On ressort le tableau, et la liste des mots
    return liste_mots, t_vectorizer.get_feature_names()

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
    fichier_save = _DOSSIERTRAVAIL + nom + '.csv'
    missing_data = data.isnull().sum(axis=0).reset_index()
    missing_data.columns = ['column_name', 'missing_count']
    missing_data['filling_factor'] = (data.shape[0]-missing_data['missing_count'])/data.shape[0]*100
    print(missing_data.sort_values('filling_factor').reset_index(drop=True))
    missing_data.sort_values('filling_factor').reset_index(drop=True).to_csv(fichier_save)

    # Transposition du dataframe de données pour l'analyse univariée
    fichier_save = _DOSSIERTRAVAIL  + 'transposition.csv'
    data_transpose = data.describe().reset_index().transpose()
    print(data_transpose)
    data_transpose.to_csv(fichier_save)

def display_topics(model, feature_names, no_top_words, vectype):
    """
    TBD
    """

    # Pour tous les topics envoyées pas le model
    for idx, topic in enumerate(model.components_):
        print("\nTopic", idx)
        listing = []

        # On rajoute chaque mot dans une liste qu'on va afficher
        for i in topic.argsort()[:-no_top_words - 1:-1]:
            listing.append(feature_names[i])

        # On affiche la liste
        print(listing)

        # Appel de la fonction pour générer le nuage de mot
        generate_wordcloud(listing, idx, vectype)

def reduction_dimension(data, limit, vectype):
    """
    Fonction de réduction de dimension
    """

    # TruncatedSVD pour réduction de dimension
    svd = TruncatedSVD(n_components=limit,
                       n_iter=5)

    svd.fit(data)

    #print(svd.explained_variance_ratio_)
    #print(svd.singular_values_)
    print("Pour", limit, "composants :", round(100*svd.explained_variance_ratio_.sum(), 2), "%")

    # Définition des axes x et y
    abs_x = range(0, limit)
    ord_y = [svd.explained_variance_ratio_[0:i].sum() for i in range(0, limit)]

    # Affichage
    plt.plot(abs_x, ord_y)
    title = 'Reduction de dimension pour ' + vectype
    plt.title(title)
    plt.tight_layout()
    plt.savefig(_DOSSIERTRAVAIL + vectype + '_reduction_', dpi=100)
    plt.show()

    # On garde la dernière valeur de i dans le fit du dessus ?
    data_reduit = svd.fit(data).transform(data)

    return data_reduit

def generate_wordcloud(data, cpt, vectype):
    """
    Simple WordCloud
    """

    # Génération du wordcloud
    wordcloud = WordCloud(background_color='black',
                          #stopwords=stopwords,
                          max_words=20,
                          max_font_size=20,
                          relative_scaling=1,
                          min_font_size=1,
                          scale=3,
                          random_state=1
                         ).generate(str(data))

    # Affichage
    fig = plt.figure(1, figsize=(6, 6))
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(wordcloud)
    plt.savefig(_DOSSIERTRAVAIL + vectype + '_wordcloud_' + str(cpt), dpi=100)
    plt.show()

def comptage(data):
    """
    Fonction qui va compter l'occurence de tous les mots
    """

    count = collections.Counter()

    stop_words = set(stopwords.words('english'))

    for sentence in data:
        sentence = word_tokenize(sentence.lower())
        for word in sentence:
            if (not word in stop_words) and word.isalpha():
                count[word] += 1

    return count

def countV():
    """
    TBD
    """

    ### ESSAI AVEC COUNTVECTORIZER
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

    # Fit du LDA crée au-dessus
    lda.fit(matrixnum_count)

    # Visualisation de la liste des mots, plus nuage de mots
    test = display_topics(lda, names_count, 10, 'lda')

    # Visualisation de la fréquence d'occurence
    visualizer = FreqDistVisualizer(features=names_count,
                                    n=25,
                                    orient='h',
                                    color='g')
    visualizer.fit(matrixnum_count)
    visualizer.poof()

    # Tentative de réduction de la dimension
    matrixnum_count_num = reduction_dimension(matrixnum_count, 2000, 'lda')

def main():
    """
    Fonction principale
    """

    # Récupération du dataset
    data = pd.read_csv(_DOSSIER + _FICHIER, error_bad_lines=False)

    # Fusion du body et du title
    data['Body'] = data['Title'] + data['Body']
    data['Tags'] = data['Tags'].str.replace("<", "")
    data['Tags'] = data['Tags'].str.replace(">", " ")

    #
    data_train = data[0:35000]
    data_test = data[35000:]

    # Nouveau dataframe qui prendra les résultats en entrée
    new_df = pd.DataFrame()

    # Données manquantes
    donnees_manquantes(data_train, "missing_data_1")

    # Suppression des balises html et des parties de code
    data_train['Body'] = [suppr_html_code(x) for x in data_train['Body']]

    # Comptage du nombre d'occurence
    cpt = comptage(data_train['Body'])

    # Liste des stop words anglais
    least_used = set([word for word in cpt if cpt[word] < 100])

    stop_words = set(stopwords.words('english')) \
                | set([word for word, freq, in cpt.most_common(100)]) \
                | least_used

    # Suppression des pluriels et des stop words
    # Rétrecissment du dataset
    new_df['Sentences'] = [fct_nltk(x, stop_words) for x in data_train['Body']]

    # On est obligé de detokenizer pour créer la matrice finale
    detokenizer = MosesDetokenizer()
    new_df['Sentences'] = [detokenizer.detokenize(x, return_str=True) for x in new_df['Sentences']]

    ### ESSAI AVEC TFIDFVECTORIZER
    # Création de la matrice finale
    matrixnum_tfidf, names_tfidf = creer_tfidfvectorizer(new_df['Sentences'])
    # Conversion de la matrice finale en dataframe pour facilité d'usage
    matrixnum_tfidf = pd.DataFrame(matrixnum_tfidf, columns=names_tfidf)

    # Run LDA
    lda = LatentDirichletAllocation(n_components=20,
                                    #max_iter=5,
                                    #learning_method='online',
                                    #learning_offset=50,
                                    random_state=0)

    # Fit du LDA crée au-dessus
    lda.fit(matrixnum_tfidf)

    # Visualisation de la liste des mots, plus nuage de mots
    display_topics(lda, names_tfidf, 15, 'lda')

    # Il faut deux matrices (distribution de proba) : documents/topic et topic/mots
    # puis multiplication des deux matrices

    # Création de la liste des tags d'origines, uniques
    liste_tags = []
    nb_tags = []

    for i in range(0, len(data_train)):
        words = word_tokenize(data_train.loc[i, 'Tags'])
        #
        nb_tags.append(len(words))

        for j in words:
            if j.isalpha() and (j not in liste_tags):
                liste_tags.append(j)

    # Probabilité d'appartanence d'une message à un topic
    df_tp1 = lda.transform(matrixnum_tfidf)
    df_tp1 = pd.DataFrame(df_tp1)

    ## PARTIE 1
    # Probabilité d'appartanence d'un mot à un topic
    df_tp3 = lda.components_ #/ nmf.components_.sum(axis=1)[:, np.newaxis]

    # Multiplication des deux matrices pour obtenir la proba documents/mots
    df_tags = df_tp1.dot(df_tp3)

    # Transformation en dataframe
    df_tags_ = pd.DataFrame(df_tags)
    df_tags_.columns = names_tfidf
    df_trans = df_tags_.T

    # Création de la matrice pour afficher les mots les plus fréquents par document
    df_plus_frequent = pd.DataFrame()

    for i in range(0, len(df_trans)):
        temp = df_trans[i].nlargest(nb_tags[i])
        temp = temp.reset_index()
        df_plus_frequent[i] = temp['index']

    df_plus_frequent = df_plus_frequent.T

    df_decompte = pd.DataFrame(columns=liste_tags)

    for i in liste_tags:
        df_decompte.loc[0, i] = data_train['Tags'].str.contains(i).sum()

    ## PARTIE 2
    # Probabilité d'appartanence d'un mot à un topic
    df_tp2 = pd.DataFrame(columns=liste_tags, index=range(0, 19))

    for i in liste_tags:
        mot = ' ' + i + ' '
        mask = data_train['Tags'].str.contains(str(mot), regex=False)
        temp = df_tp1[mask]

        for j in df_tp1.columns:
            df_tp2.loc[j, i] = temp[j].sum()

    df_tp2 = df_tp2.astype(float)
    m = df_tp2.sum(axis=1)[:, np.newaxis]

    for i in range(0, 20):
        df_tp2.loc[i] = df_tp2.loc[i] / m[i]

    # Multiplication des deux matrices pour obtenir la proba documents/mots
    df_tags = df_tp1.dot(df_tp2)

    # Transformation en dataframe
    df_trans = df_tags.T.astype(float)

    # Création de la matrice pour afficher les mots les plus fréquents par document
    df_prevision = []

    for i in range(0, len(df_tags)):
        temp = df_trans[i].nlargest(nb_tags[i])
        temp = temp.reset_index()
        df_prevision.append(temp['index'])

    df_prevision = pd.DataFrame(df_prevision).reset_index(drop=True)

    count_tag = 0

    for i in range(0, len(data_train)):
        liste_tags = word_tokenize(data_train.loc[i, 'Tags'])

        for tag in liste_tags:
            for j in range(0, 10):
                if tag == df_prevision.loc[i, j]:
                    count_tag = count_tag + 1

    print(round(count_tag/df_prevision.count().sum()*100, 3), '%')

    # Visualisation de la fréquence d'occurence
    ## voir la fréquence minimale
    visualizer = FreqDistVisualizer(features=names_tfidf,
                                    n=25,
                                    orient='v',
                                    color='g')
    visualizer.fit(matrixnum_tfidf)
    visualizer.poof()

    # Tentative de réduction de la dimension
    matrixnum_tfidf_num = reduction_dimension(matrixnum_tfidf, 500, 'lda')

    # Supervisé
    # one hot encoding sur un set de tag assez fréquent
    # one versus rest classifier
    # sklearn multi label

def test():

    for i in data_train['Tags']:
        for j in word_tokenize(i):
            for k in df_plus_frequent.index:
                if k == j :
                    print(i, j, k)

    mask = data_train['Tags'].str.contains('python')

    for i in temp:
        print("topic", i, temp[i].sum())

    df_tp4 = pd.DataFrame(columns=liste_tags, index=range(0,19))

    for i in liste_tags:
        mask = data_train['Tags'].str.contains(str(i))
        temp = df_tp1[mask]
        for j in df_tp1.columns:
            df_tp4.loc[j, i] = temp[j].sum()

    df_decompte = pd.DataFrame(columns=liste_tags)

    for i in liste_tags:
        df_decompte.loc[0, i] = data_train['Tags'].str.contains(i).sum()


# prendre tous les doucments qui ont python dans un tag
