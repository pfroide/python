# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm as cm
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn import cluster, metrics, manifold
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn import decomposition

# =============================================================================
# Après le mentorat de Thierry :
# Voir si les fonctions que j'ai prises sur kaggle ne sont pas de trop.
# Ne pas prendre en compte dans un premier temps les mots-clefs car trop nombreux.
# =============================================================================

# cleaner le dataset
# mettre les valeurs que je veux
# calculer les distances 

# pca
# variation du facteur  silhouette en fonction de nb de clusters
# sklearn.metrics.silhouette_score

# voir le r2 pour la correlation
# voir ave le scatter plot

# indice de popularité du film dans un deuxième temps pour départager ceux qui pourraient etre dans un meme cluster

# Lieu où se trouve le fichier
_FICHIER = 'C:\\Users\\Toni\\Desktop\\movie_metadata.csv'
_DOSSIERTRAVAIL = 'C:\\Users\\Toni\\python\\python\\Projet_3\\images'

# function that extract statistical parameters from a grouby objet
def get_stats(param):
    """
    TBD
    """
    return {'min':param.min(),
            'max':param.max(),
            'count': param.count(),
            'mean':param.mean()
           }

def count_word(data, ref_col, liste):
    """
    TBD
    """
    keyword_count = dict()
    
    for s in liste:
        keyword_count[s] = 0
    
    for liste_keywords in data[ref_col].str.split('|'):
        if isinstance(liste_keywords, float) and pd.isnull(liste_keywords):
            continue
        for s in [s for s in liste_keywords if s in liste]:
            if pd.notnull(s):
                keyword_count[s] += 1

    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []
    
    for k, v in keyword_count.items():
        keyword_occurences.append([k, v])
    
    keyword_occurences.sort(key=lambda x: x[1], reverse=True)
    
    return keyword_occurences, keyword_count

def afficher_plot(type_donnee, trunc_occurences):
    """
    TBD
    """
    fichier_save = _DOSSIERTRAVAIL + '\\' + type_donnee

    words = dict()

    for s in trunc_occurences:
        words[s[0]] = s[1]

    plt.figure(figsize=(15, 10))
    y_axis = [i[1] for i in trunc_occurences]
    x_axis = [k for k, i in enumerate(trunc_occurences)]
    x_label = [i[0] for i in trunc_occurences]
    plt.xticks(rotation=90, fontsize=10)
    plt.xticks(x_axis, x_label)

    plt.yticks(fontsize=10)
    plt.ylabel("Nb. of occurences", fontsize=18, labelpad=10)

    plt.bar(x_axis, y_axis, align='center', color='b')

    plt.savefig(fichier_save, dpi=100)

    plt.title(type_donnee + " popularity", fontsize=25)
    plt.show()

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

def correlation_matrix(data):
    """
        Fonction qui permet de créer une visualisation du lien entre les
        variables 2 à 2
    """
    # Calcule de la matrice
    corr = data.corr()
    cmap = cm.get_cmap('jet', 30)

    # Taille de la figure
    plt.figure(figsize=(10, 10))
    # Création du type d'image
    cax = plt.imshow(data.corr(), interpolation="nearest", cmap=cmap)
    plt.grid(True)

    # Libellés sur les axes
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=15)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=15)

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    plt.colorbar(cax, ticks=[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    plt.show()

def transpose_bool(data):
    
    # énumaration des genres
    listing = comptabiliser(data, 'genres')
    
    for mot, compte in listing:
        data[mot] = pd.Series(((1 if mot in data['genres'][i] else 0) for i in range(5043)), index=data.index)
    
    # On supprime les #NA
    data['plot_keywords'].fillna('vide', inplace=True)
    
    # énumaration des mot-clefs
    listing = comptabiliser(data, 'plot_keywords')
    
    for mot, compte in listing:
        if compte > 30:
            data[mot] = pd.Series(((1 if mot in data['plot_keywords'][i] else 0) for i in range(5043)), index=data.index)
    
    # Suppression de la colonne "vide"
    del data['vide']
    
    # etudier l'impact des n premiers

#def affichage_nearest_neighbours(data, nom_film):
#
#     # Suprresion de ce qui n'est pas chiffré
#    datanum = data.copy()
#    datanum = datanum.drop(['color', 'director_name', 'actor_1_name', 'genres', 'movie_title', 'actor_2_name', 'actor_3_name'], axis=1)
#    datanum = datanum.drop(['plot_keywords', 'movie_imdb_link', 'language', 'country', 'content_rating'], axis=1)
#
#    # Suppresion des NA
#    datanum.fillna(0, inplace=True)
#
##    # Scale des données obligatoire avant la réduction des dimensions
##    std_scale = preprocessing.StandardScaler().fit(datanum)
##    X_scaled = std_scale.transform(datanum)
##    
##    # Réduction t-Sne
##    print("Computing t-SNE embedding")
##    tsne = manifold.TSNE(n_components=2, perplexity=50, n_iter=500)
##    
##    nbrs = NearestNeighbors(n_neighbors=31, algorithm='auto', metric='euclidean').fit(datanum)    
##    distances, indices = nbrs.kneighbors(datanum)
#    
#    
#    #X = df_new.as_matrix(datanum)
#    X = datanum
#    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean').fit(X)
#
#    distances, indices = nbrs.kneighbors(X) 
#
#    print(distances)
#    print(indices)

def affichage_kmeans(data):
    
    # Suprresion de ce qui n'est pas chiffré
    datanum = data.copy()
    datanum = datanum.drop(['color', 'director_name', 'actor_1_name', 'genres', 'movie_title', 'actor_2_name', 'actor_3_name'], axis=1)
    datanum = datanum.drop(['plot_keywords', 'movie_imdb_link', 'language', 'country', 'content_rating'], axis=1)

    # Suppresion des NA
    datanum.fillna(0, inplace=True)

    # Scale des données obligatoire avant la réduction des dimensions
    std_scale = preprocessing.StandardScaler().fit(datanum)
    X_scaled = std_scale.transform(datanum)
    
    # Réduction t-Sne
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, perplexity=50, n_iter=500)
    
    for i in range(50,51):
        # On fait i clusters avec les données scalées.
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X_scaled)
        # Nouvelle colonne avec les conclusions de kmeans
        datanum['labels'] = kmeans.labels_
    
        # perform t-SNE embedding
        x_tsne = tsne.fit_transform(X_scaled)

        # plot the result
        vis_x = x_tsne[:, 0]
        vis_y = x_tsne[:, 1]
    
        # figure de prédiction
        plt.figure(figsize=(10, 6))
        plt.scatter(vis_x, vis_y, c=datanum['labels'], cmap=plt.cm.get_cmap("jet", i))
        plt.colorbar(ticks=range(i))
        plt.grid('on')
        Titre = 'Affichage du clustering déduit par le Kmeans n=%s ' % i
        plt.title(Titre)
        plt.show()
    
    # On remets les titres pour y voir plus clair
    datanum['movie_title'] = data['movie_title']

def main():
    """
    TBD
    """
    # On charge le dataset
    data = pd.read_csv(_FICHIER)

    # Suppresion de doublons
    data.drop_duplicates(subset=['movie_title', 'actor_1_name', 'director_name'], keep='first')

    print("Données manquantes")
    # nombre de valeurs "NaN"
    # .isnull().sum() = nombre par ligne
    # .isnull().sum().sum() = nombre total
    # Compte les données manquantes par colonne
    missing_data = data.isnull().sum(axis=0).reset_index()

    # Change les noms des colonnes
    missing_data.columns = ['column_name', 'missing_count']

    # Crée une nouvelle colonne et fais le calcul en pourcentage des données
    # manquantes
    missing_data['filling_factor'] = (data.shape[0]-missing_data['missing_count'])/data.shape[0]*100

    # Classe et affiche
    missing_data.sort_values('filling_factor').reset_index(drop=True)

    # Transposition du dataframe de données pour l'abalyse univariée
    fichier_save = _DOSSIERTRAVAIL + '\\' + 'transposition.csv'
    data_transpose = data.describe().reset_index().transpose()
    print (data_transpose)
    data_transpose.to_csv(fichier_save)
    
    # Matrice de correlation
    correlation_matrix(data)
    
    data.fillna(0, inplace=True)
    
    regression = []
    
    colon1 = 'gross'
    
    for nom_colonne in data:    
        colon2 = nom_colonne

        if (data[nom_colonne].dtype == 'float' or data[nom_colonne].dtype == 'int64') and colon2 != 'gross' :
            # Calcul d'une regression linéaire
            regr = linear_model.LinearRegression()
            regr.fit(data[colon1].values.reshape(-1, 1), data[colon2].values.reshape(-1, 1))
        
            # Affichage de la variances : On doit être le plus proche possible de 1
            print('Regression sur les deux colonnes :', colon1, colon2)
            print('Score : %.2f' % np.corrcoef(data[colon1], data[colon2])[1, 0])
            regression.append(np.corrcoef(data[colon1], data[colon2])[1, 0])

    # Création de la database avec tous les noms d'acteurs car ils sont sur
    # 3 colonnes différentes
    db_names = []
    db_names.extend(data['actor_1_name'])
    db_names.extend(data['actor_2_name'])
    db_names.extend(data['actor_3_name'])
    data_names = pd.DataFrame(db_names, columns=['name'])

    # compter tous genres des films
    genre_list = comptabiliser(data, 'genres')

    # compter tous languages des films
    language_list = comptabiliser(data, 'language')

    # compter tous les pays des films
    country_list = comptabiliser(data, 'country')

    # compter tous les ratings
    rating_list = comptabiliser(data, 'content_rating')

    # compter tous mots-clefs des films
    keywords_list = comptabiliser(data, 'plot_keywords')

    # compter tous les directeurs de films
    directors_list = comptabiliser(data, 'director_name')

    # compter tous les acteurs de films
    actors_list = comptabiliser(data_names, 'name')

    # Affichages
    afficher_plot('genres', genre_list[0:50])
    afficher_plot('keywords', keywords_list[0:50])
    afficher_plot('directors', directors_list[0:50])
    afficher_plot('actors', actors_list[0:50])
    afficher_plot('languages', language_list[0:50])
    afficher_plot('countrys', country_list[0:50])
    afficher_plot('ratings', rating_list[0:50])

    std_scale = preprocessing.StandardScaler().fit(datanum)
    X_scaled = std_scale.transform(datanum)

#    pca = decomposition.PCA(n_components=5)
#    pca.fit(X_scaled)
#
#    print(pca.explained_variance_ratio_)
#    print(pca.explained_variance_ratio_.sum())
#
#    # projeter X sur les composantes principales
#    X_projected = pca.transform(X_scaled)
#
#    import matplotlib.cm as cm
#    m = cm.ScalarMappable(cmap=cm.jet)
#    m.set_array(X_projected)
#    plt.colorbar(m)
#    
#    # afficher chaque observation
#    plt.scatter(X_projected[:, 0], X_projected[:, 1], c=data.get('imbd_score'))
#    #plt.xlim([-5.5, 5.5])
#    #plt.ylim([-4, 4])
#    plt.colorbar(m)
#
#    # Pour mieux comprendre ce que capture ces composantes principales, 
#    # nous pouvons utiliser pca.components_, qui nous donne les coordonnées 
#    # des composantes principales dans l'espace initial (celui à 10 variables).
#    # Nous allons afficher, pour chacune des 10 performances, 
#    # un point dont l'abscisse sera sa contribution à la première PC 
#    #   et l'ordonnée sa contribution à la deuxième PC.
#    pcs = pca.components_
#    
#    for i, (x, y) in enumerate(zip(pcs[0, :], pcs[1, :])):
#        # Afficher un segment de l'origine au point (x, y)
#        plt.plot([0, x], [0, y], color='k')
#        # Afficher le nom (data.columns[i]) de la performance
#        plt.text(x, y, data.columns[i], fontsize='8')
#    
#    # Afficher une ligne horizontale y=0
#    plt.plot([-0.7, 0.7], [0, 0], color='grey', ls='--')
#    
#    # Afficher une ligne verticale x=0
#    plt.plot([0, 0], [-0.7, 0.7], color='grey', ls='--')
#    
#    plt.xlim([-0.7, 0.7])
#    plt.ylim([-0.7, 0.7])
#
#
#
#
#    # Affichage des décades
#    data['decade'] = data['title_year'].apply(lambda x: ((x)//10)*10)
#
#    # Creation of a dataframe with statitical infos on each decade:
#    test = data['title_year'].groupby(data['decade']).apply(get_stats).unstack()
#
#    sizes = test['count'].values / (data['title_year'].count()) * 100
#
#    # pour le camembert
#    # Attention car y'a aussi ceux qui n'ont pas de dates.
#    explode = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
#
#    # affichage du camembert
#    plt.pie(sizes,
#            explode=explode,
#            labeldistance=1.2,
#            labels=round(test['min'], 0),
#            shadow=False,
#            startangle=0,
#            autopct=lambda x: '{:1.0f}%'.format(x) if x > 5 else '')
#
#    # Liste des noms complets à analyser
#    alphabet = []
#    alphabet.append('num_critic_for_reviews')
#    alphabet.append('num_user_for_reviews')
#    alphabet.append('duration')
#    alphabet.append('gross')
#    alphabet.append('budget')
#    alphabet.append('imdb_score')
#    alphabet.append('movie_facebook_likes')
#    alphabet.append('cast_total_facebook_likes')
#    alphabet.append('num_voted_users')
#
#     # Affichage des imdb_score par 10
#    data['imdb_score10'] = data['imdb_score'].apply(lambda x: round(x, 0))
#
#    # Creation of a dataframe with statitical infos on each decade:
#    test = data['imdb_score'].groupby(data['imdb_score10']).apply(get_stats).unstack()
#    sizes = test['count'].values / (data['imdb_score'].count()) * 100
#
#    # pour le camembert
#    explode = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
#
#    # affichage du camembert
#    plt.pie(sizes,
#            explode=explode,
#            labeldistance=1.2,
#            labels=round(test['min'], 0),
#            shadow=False,
#            startangle=0,
#            autopct=lambda x: '{:1.0f}%'.format(x) if x > 5 else '')
#
#
#    data['budget10'] = data['budget'].apply(lambda x: ((x)//1000000))
#
#    # Creation of a dataframe with statitical infos on each decade:
#    test = data['budget'].groupby(data['budget10']).apply(get_stats).unstack()
#    sizes = test['count'].values / (data['budget'].count()) * 100
#
#    # affichage du camembert
#    plt.pie(sizes,
#            labeldistance=1.2,
#            labels=round(test['min'], 0),
#            shadow=False,
#            startangle=0,
#            autopct=lambda x: '{:1.0f}%'.format(x) if x > 5 else '')
#    
#    
#    df_filling = data.copy(deep=True)
#    missing_year_info = df_filling[df_filling['title_year'].isnull()][[
#            'director_name','actor_1_name', 'actor_2_name', 'actor_3_name']]
#    missing_year_info[:]
#    
#    df_filling.iloc[177]
