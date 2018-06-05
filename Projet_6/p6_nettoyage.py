# -*- coding: utf-8 -*-
"""
Created on Thu May 17 21:51:03 2018

@author: Toni
"""

# On importe les librairies dont on aura besoin pour ce tp
import sys
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
    t_vectorizer = TfidfVectorizer(min_df = 0.01)

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

def main():
    """
    Fonction principale
    """

    # Récupération du dataset
    data = pd.read_csv(_DOSSIER + _FICHIER, error_bad_lines=False)

    #
    data = data[0:35000]

    # Fusion du body et du title
    data['Body'] = data['Title'] + data['Body']

    # Nouveau dataframe qui prendra les résultats en entrée
    new_df = pd.DataFrame()

    # Données manquantes
    donnees_manquantes(data, "missing_data_1")

    # Suppression des balises html et des parties de code
    data['Body'] = [suppr_html_code(x) for x in data['Body']]

    # Comptage du nombre d'occurence
    cpt = comptage(data['Body'])

    # Liste des stop words anglais
    least_used = set([word for word in cpt if cpt[word] < 10])

    stop_words = set(stopwords.words('english')) \
                | set([word for word, freq, in cpt.most_common(100)]) \
                | least_used

    # Suppression des pluriels et des stop words
    # Rétrecissment du dataset
    new_df['Sentences'] = [fct_nltk(x, stop_words) for x in data['Body']]

    # On est obligé de detokenizer pour créer la matrice finale
    detokenizer = MosesDetokenizer()
    new_df['Sentences'] = [detokenizer.detokenize(x, return_str=True) for x in new_df['Sentences']]

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
    display_topics(lda, names_count, 20, 'lda')

    # Visualisation de la fréquence d'occurence
    visualizer = FreqDistVisualizer(features=names_count,
                                    n=25,
                                    orient='v',
                                    color='g')
    visualizer.fit(matrixnum_count)
    visualizer.poof()

    # Tentative de réduction de la dimension
    matrixnum_count_num = reduction_dimension(matrixnum_count, 2000, 'lda')

    ### ESSAI AVEC TFIDFVECTORIZER
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

    # Fit du NMF créé au-dessus
    nmf.fit(matrixnum_tfidf)

    # Visualisation de la liste des mots, plus nuage de mots
    display_topics(nmf, names_tfidf, 20, 'nmf')

    # Visualisation de la fréquence d'occurence
    ## voir la fréquence minimale
    visualizer = FreqDistVisualizer(features=names_tfidf,
                                    n=25,
                                    orient='v',
                                    color='g')
    visualizer.fit(matrixnum_tfidf)
    visualizer.poof()

    # Tentative de réduction de la dimension
    matrixnum_tfidf_num = reduction_dimension(matrixnum_tfidf, 500, 'nmf')
