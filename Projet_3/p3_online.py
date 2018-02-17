"""
Created on Wed Feb 14 20:22:34 2018

@author: Toni
"""
# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from flask import Flask
import json

# Lieu où se trouve le fichier
_DOSSIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\'
_FICHIERDATA = _DOSSIER + 'p3_bdd_clean_v2.csv'
_FICHIERDATANUM = _DOSSIER + 'p3_datanum.csv' 

app = Flask(__name__)

#@app.route('/recommand/<id_film>')
@app.route('/<id_film>')

def recommand(id_film): 

    data = pd.read_csv(_FICHIERDATA, encoding="ISO-8859-1")
    datanum = pd.read_csv(_FICHIERDATANUM, encoding="ISO-8859-1")
    data = data.replace({'\xa0': ''}, regex=True)
    del data['Unnamed: 0']
    del datanum['Unnamed: 0']

    #print(recommandation(datanum, data, id_film))

    return recommandation(datanum, data, id_film)

def recommandation(datanum, data, id_film):
    """
    TBD
    """

    #
    texte_final = ''

    #
    if isinstance(id_film, str):

        # Rajout des noms
        datanum['movie_title'] = data['movie_title']

        # Recherche des données du film
        data_film = datanum.loc[datanum['movie_title'].str.contains(id_film)]

        # Quel est le titre recherché
        mask = datanum['movie_title'].str.contains(id_film)
        titre = data['movie_title'][mask]

        # Astuce pour avoir un string au bon format
        for i in titre[-1:]:
            titre_fin = i

        # Suppression de la colonne non-chifrée
        del data_film['movie_title']
        del datanum['movie_title']

    else:
        data_film = datanum.loc[int(id_film)].values.reshape(1, -1)
        # Quel est le titre recherché
        titre_fin = data['movie_title'].loc[int(id_film)]

    data_film = np.array(data_film)

    if data_film.size > 0:

        # Nom du film retenu
        texte_final = 'Titre retenu : ' + titre_fin + '<br>'

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
            if str(data.loc[i]['movie_title']) in titre_fin:
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
                          'num_user_for_reviews',
                          'num_voted_users',
                          'movie_facebook_likes']

        for colon in second_df:
            if colon not in liste_criteres:
                del second_df[colon]

        # On enlève les Nan
        second_df.fillna(0, inplace=True)

        # Tester la popularité
        reponse = pop(second_df, data, datanum, id_film)
        texte_final = texte_final + reponse

    else:
        texte_final = "Le film recherché n'existe pas."

    return texte_final

def pop(second_df, data, datanum, id_film):
    """
    TBD
    """

    # Indice de popularité simpliste
    min_max_scaler = preprocessing.MinMaxScaler()
    second_df[['cast_total_facebook_likes',
               'imdb_score',
               'num_user_for_reviews',
               'num_voted_users',
               'movie_facebook_likes']] = min_max_scaler.fit_transform(second_df[['cast_total_facebook_likes', 'imdb_score', 'num_user_for_reviews', 'num_voted_users', 'movie_facebook_likes']])

    second_df['score'] = second_df['cast_total_facebook_likes'] + second_df['imdb_score'] + second_df['movie_facebook_likes'] + second_df['num_user_for_reviews'] + second_df['num_voted_users']

    # Score du film
    score = sum(second_df.score[second_df.movie_title == id_film])

    # Calcul de la valeur absolue du score pour voir la différence avec les autres films
    second_df['score'] = abs(second_df['score']-score)
    second_df = second_df.sort_values(by='score', ascending=True)
    
    # 5 résultats
    second_df = second_df[0:5]
    
    # Résultats en dictionnaire
    #dico_resultats = {}

    # 5 résultats les plus proches sans prendre le film lui-même qui est en case 0
    #for i in range(0,5):
    #    dico_resultats[str(second_df.index[i])] = second_df['movie_title'][second_df.index[i]]

    # 5 résultats les plus proches sans prendre le film lui-même
    dico = {"_results" :[{'id':int(key),"name":value} for key, value in second_df['movie_title'].items()]}
    
    json_dico = json.dumps(dico, indent=4, separators=(',', ': '))

    # print(dico_resultats)
    return(json_dico)
