# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 20:22:34 2018

@author: Toni
"""
# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from flask import Flask  # pip install flask

# Lieu où se trouve le fichier
_FICHIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\p3_bdd_clean_v2.csv'
_DOSSIERTRAVAIL = 'C:\\Users\\Toni\\python\\python\\Projet_3\\images'

app = Flask(__name__)

@app.route("/")

def hello():
    return "  World!"

if __name__ == "__main__":
    app.run()

def count_word(data, ref_col, liste):
    """
    TBD
    """
    keyword_count = dict()

    for word in liste:
        keyword_count[word] = 0

    for liste_keywords in data[ref_col].str.split('|'):
        if isinstance(liste_keywords, float) and pd.isnull(liste_keywords):
            continue
        for word in [word for word in liste_keywords if word in liste]:
            if pd.notnull(word):
                keyword_count[word] = keyword_count[word] + 1

    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []

    for k, v in keyword_count.items():
        keyword_occurences.append([k, v])

    keyword_occurences.sort(key=lambda x: x[1], reverse=True)

    return keyword_occurences, keyword_count

def comptabiliser(data, valeur_cherchee):
    """
    TBD
    """
    # compter tous les genres différents
    listing = set()

    for word in data[valeur_cherchee].str.split('|').values:
        if isinstance(word, float):
            continue
        listing = listing.union(word)

    # compter le nombre d'occurence de ces genres
    listing_compte, dum = count_word(data, valeur_cherchee, listing)

    return listing_compte

def transpose_bool(data, colon, limite):
    """
    TBD
    """

    # On supprime les #NA
    data[colon].fillna('vide', inplace=True)

    # énumaration des genres
    listing = comptabiliser(data, colon)

    p = 0

    for mot, compte in listing:
        if p < limite:
            if mot not in data:
                data[mot] = pd.Series(((1 if mot in data[colon][i] else 0) for i in range(len(data[colon]))), index=data.index)
            else:
                data[mot] = pd.Series(((1 if (np.logical_or(data[mot].item == 1, mot in data[colon][i])) else 0) for i in range(len(data[colon]))), index=data.index)
        else:
            return p
        p = p+1

    return p

def recommandation(datanum, data, id_film):
    """
    TBD
    """
    #
    if isinstance(id_film, int):
        data_film = datanum.loc[id_film].values.reshape(1, -1)
        # Quel est le titre recherché
        titre = data['movie_title'].loc[id_film]
        titre_fin = titre
        print('Titre retenu : ', titre)
    else:
        
        # Rajout des noms
        datanum['movie_title'] = data['movie_title']

        # Recherche des données du film
        data_film = datanum.loc[datanum['movie_title'].str.contains(id_film)]

        # Quel est le titre recherché
        mask = datanum['movie_title'].str.contains(id_film)
        titre = data['movie_title'][mask]
        titre_fin = str(titre.values[-1:])
        print('Titre retenu : ', titre.values[-1:])

        # Suppression de la colonne non-chifrée
        del data_film['movie_title']
        del datanum['movie_title']

    data_film = np.array(data_film)

    if data_film.size > 0:

        # configuration du knn
        neigh = NearestNeighbors(n_neighbors=20,
                                 algorithm='auto',
                                 metric='euclidean'
                                )

        # knn
        neigh.fit(datanum)
        indices = neigh.kneighbors(data_film)

        indice_supp = []

        for i in indices[1][-1]:
            #print(data.loc[i]['movie_title'])
            if ((str(data.loc[i]['movie_title']) in titre_fin)):
                indice_supp.append(i)
            elif (titre_fin in (str(data.loc[i]['movie_title']))):
                indice_supp.append(i)

        #
        for i in indices[1]:
            second_df = data.loc[i] 
        
        for i in indice_supp:
            second_df = second_df.drop([i])
        
        # Création du second dataset pour tester la popularité
        liste_criteres = ['movie_title',
                          'cast_total_facebook_likes',
                          'imdb_score',
                          'movie_facebook_likes']

        for colon in second_df:
            if colon not in liste_criteres:
                del second_df[colon]

        # On enlève les Nan
        second_df.fillna(0, inplace=True)

        # Tester la popularité
        pop(second_df, data, datanum, id_film)

    else:
        print("Le film recherché n'existe pas.")

def pop(second_df, data, datanum, id_film):
    """
    TBD
    """

    # Indice de popularité simpliste
    min_max_scaler = preprocessing.MinMaxScaler()
    second_df[['cast_total_facebook_likes',
               'imdb_score',
               'movie_facebook_likes']] = min_max_scaler.fit_transform(second_df[['cast_total_facebook_likes', 'imdb_score', 'movie_facebook_likes']])
    
    second_df['score'] = second_df['cast_total_facebook_likes'] + second_df['imdb_score'] + second_df['movie_facebook_likes']

    # Score du film
    score = sum(second_df.score[second_df.movie_title == id_film])

    # Calcul de la valeur absolue du score pour voir la différence avec les autres films
    second_df['score'] = abs(second_df['score']-score)
    second_df = second_df.sort_values(by='score', ascending=True)
    
    # 5 résultats les plus proches sans prendre le film lui-même qui est en case 0
    res = second_df['movie_title'][1:6]

    # Résultats
    print(res)

    # Petite vérification des mot-clefs associés
    for i in range(len(res)):
        print(data.loc[res.index[i]]['movie_title'])
        for j in datanum:
            if datanum.loc[res.index[i]][j] == 1:
                print(j)
        print()
        # datanum.loc[n][datanum.loc[n] == 1]

def second(second_df):
    """
    TBD
    """

    liste_criteres = ['cast_total_facebook_likes',
                      'num_user_for_reviews',
                      'imdb_score',
                      'movie_facebook_likes']

    for colon in second_df:
        if colon not in liste_criteres:
            del second_df[colon]

def premain():
    """
    TBD
    """
    
    #data_test.set_index("Robert de Niro", inplace=True)
    # On charge le dataset
    data = pd.read_csv(_FICHIER, encoding="ISO-8859-1")

    # Suppression de la colonne inutile
    del data['Unnamed: 0']

    # Suppression des espaces en fin de chaine de caractères
    data['movie_title'] = data['movie_title'].replace({'\xa0': ''}, regex=True)
    data['movie_title'] = data['movie_title'].str.strip()

    datanum = data.copy()
    datanum.describe()

    # Données manquantes
    missing_data = datanum.isnull().sum(axis=0).reset_index()
    missing_data.columns = ['column_name', 'missing_count']
    missing_data['filling_factor'] = (datanum.shape[0]-missing_data['missing_count'])/datanum.shape[0]*100
    missing_data.sort_values('filling_factor').reset_index(drop=True)

    # Transposition en 0 et 1 des valeurs non-numériques
    liste_criteres = ['actor_1_name',
                      'actor_2_name',
                      'actor_3_name',
                      'genres',
                      'content_rating',
                      'director_name']

    for critere in liste_criteres:
        num = transpose_bool(datanum, critere, 100)
        print("Nombre : ", num, "\t", critere)

    # Suprresion de ce qui n'est pas chiffré
    datanum = datanum.drop(['color', 'director_name', 'actor_1_name', 'genres', 'movie_title', 'actor_2_name', 'actor_3_name'], axis=1)
    datanum = datanum.drop(['plot_keywords', 'movie_imdb_link', 'language', 'country', 'content_rating'], axis=1)

    # Suprresion de ce qui n'est pas chiffré #2
    datanum = datanum.drop(['num_critic_for_reviews', 'director_facebook_likes', 'actor_3_facebook_likes'], axis=1)
    datanum = datanum.drop(['num_user_for_reviews', 'actor_1_facebook_likes', 'actor_2_facebook_likes'], axis=1)
    datanum = datanum.drop(['aspect_ratio', 'num_voted_users', 'cast_total_facebook_likes'], axis=1)
    datanum = datanum.drop(['title_year', 'gross', 'duration', 'budget', 'imdb_score', 'movie_facebook_likes', 'facenumber_in_poster'], axis=1)