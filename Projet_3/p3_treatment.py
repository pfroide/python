# -*- coding: utf-8 -*-

# On importe les librairies dont on aura besoin pour ce tp
from __future__ import print_function

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm as cm
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn import cluster, metrics, manifold
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.metrics import silhouette_samples, silhouette_score
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs

import matplotlib

# =============================================================================
# Après le mentorat de Thierry :
# Voir si les fonctions que j'ai prises sur kaggle ne sont pas de trop.
# Ne pas prendre en compte dans un premier temps les mots-clefs car trop nombreux.
# =============================================================================

# etudier l'impact des n premiers

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
_FICHIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\p3_bdd_clean_v2.csv'
_DOSSIERTRAVAIL = 'C:\\Users\\Toni\\python\\python\\Projet_3\\images'

def illustration(data, range_n_clusters):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    #print(__doc__)

    # Scale des données obligatoire avant la réduction des dimensions
    std_scale = preprocessing.StandardScaler().fit(data)
    X = std_scale.transform(data)
    
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(36, 14)
    
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
    
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
# =============================================================================
#         # 2nd Plot showing the actual clusters formed
#         colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
#         ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
#                     c=colors, edgecolor='k')
#     
#         # Labeling the clusters
#         centers = clusterer.cluster_centers_
#         # Draw white circles at cluster centers
#         ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
#                     c="white", alpha=1, s=200, edgecolor='k')
#     
#         for i, c in enumerate(centers):
#             ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
#                         s=50, edgecolor='k')
#     
#         ax2.set_title("The visualization of the clustered data.")
#         ax2.set_xlabel("Feature space for the 1st feature")
#         ax2.set_ylabel("Feature space for the 2nd feature")
# =============================================================================
    
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
    
        plt.show()

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

def affichage_kmeans(datanum, vmin, vmax, step):

    listing = dict()
    distortions = []
    
    # Scale des données obligatoire avant la réduction des dimensions
    std_scale = preprocessing.StandardScaler().fit(datanum)
    X_scaled = std_scale.transform(datanum)
    
    # Réduction t-Sne
    #print("Computing t-SNE embedding")
    #tsne = manifold.TSNE(n_components=2, perplexity=50, n_iter=500)
    
    cluster_range = range(vmin,vmax+1, step)
    
    for i in cluster_range:
        # On fait i clusters avec les données scalées.
#        kmeans = KMeans(n_init=50,
#                        n_clusters=i,
#                        init="k-means++",
#                        max_iter=1000,
#                        verbose=0,
#                        algorithm="auto")

        kmeans = KMeans(n_clusters=i, random_state=10)

        kmeans.fit(X_scaled)
        # Nouvelle colonne avec les conclusions de kmeans
        datanum['labels'] = kmeans.labels_
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        print("For n_clusters =", i,
              "The average silhouette_score is :", silhouette_avg)
        
        #distortions.append(sum(np.min(cdist(X_scaled, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X_scaled.shape[0])
        distortions.append(kmeans.inertia_)
        listing[i] = silhouette_avg
    
    # Données du graphique du coude
    clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": distortions } )
    print(clusters_df)
    
    # Graphique du coude
    plt.figure(figsize=(15, 15))
    plt.plot(cluster_range, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    fichier_save = _DOSSIERTRAVAIL + '\\' + 'coude' 
    plt.savefig(fichier_save, dpi=100)
    plt.show()
    
    return listing, clusters_df

def transpose_bool(data, colon, limite):
    
    # On supprime les #NA
    data[colon].fillna('vide', inplace=True)
    
    # énumaration des genres
    listing = comptabiliser(data, colon)

    p=0    
    for mot, compte in listing:
        if p<limite:
            data[mot] = pd.Series(((1 if mot in data[colon][i] else 0) for i in range(len(data[colon]))), index=data.index)
        else:
            return 0
        p=p+1
        
    # Suppression de la colonne "vide"
    #del data['vide']

def histogramme(data, colon):
    """
        Note : La première colonne et la dernière ont un " caché
    """

    #fichier_save = _DOSSIERTRAVAIL + '\\' + 'histogram_' + colon

    #steps = (max(data[colon])-min(data[colon]))/100
    #bin_values = np.arange(start=min(data[colon]), stop=max(data[colon]), step=steps)
    plt.figure(figsize=(10, 6))
    plt.xlabel('Valeurs')
    plt.ylabel('Décompte')
    titre = 'Histogramme ' + colon
    plt.title(titre)
    # plt.hist(data[colon], bins=bin_values)
    # Test sans les valeurs NaN
    plt.hist(data[colon][np.isfinite(data[colon])], bins=100)
    #plt.savefig(fichier_save, dpi=100)

def tester_moteur(datanum, titre):
    print(datanum['labels'][datanum['movie_title'].str.contains(titre)])
    print(datanum['movie_title'][datanum['movie_title'].str.contains(titre)])

def test(data):
    
    import seaborn as sn
    from sklearn.preprocessing import StandardScaler
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    cmap = sn.cubehelix_palette(as_cmap=True, rot=-.3, light=1)
    sn.clustermap(X_scaled, cmap=cmap, linewidths=.5)

def trouver_films_nom(data, titre):
    
    # liste_films = []
    
    label = int(data['labels'][data['movie_title'].replace({'\xa0': ''}, regex=True) == titre])
    
    print(data['movie_title'][data['labels'] == label])
    #print(data['movie_title'].where(data['labels'] == label))
    
    # Tentatives de conversion de chaines de caractères
    # p=data.where(data['director_name'] == 'James Cameron')
    # p=p.where(p['movie_title'] == 'The Terminator')
    # x=data['movie_title'].replace({'\xa0': ''}, regex=True)

def trouver_films_id(data, film_id):
    
    # liste_films = []
    label = data.index(film_id)
    
    print(data['movie_title'][data['labels'] == label])
    
def main():
    """
    TBD
    """
    # On charge le dataset
    data = pd.read_csv(_FICHIER, encoding = "ISO-8859-1")
    del data['Unnamed: 0']
    
    datanum = data.copy()
    datanum.describe()
    
    # Données manquantes
    missing_data = datanum.isnull().sum(axis=0).reset_index()
    missing_data.columns = ['column_name', 'missing_count']
    missing_data['filling_factor'] = (datanum.shape[0]-missing_data['missing_count'])/datanum.shape[0]*100
    missing_data.sort_values('filling_factor').reset_index(drop=True)

    # Pour les noms d'acteurs    
    #db_names = []
    #db_names.extend(data['actor_1_name'])
    #db_names.extend(data['actor_2_name'])
    #db_names.extend(data['actor_3_name'])
    #datanames = pd.DataFrame(db_names, columns=['name'])

    #transpose_bool(datanum, 'language', 50)
    #transpose_bool(datanum, 'country')    
    transpose_bool(datanum, 'genres', 50)
    transpose_bool(datanum, 'plot_keywords', 50)
    transpose_bool(datanum, 'color', 50)
    transpose_bool(datanum, 'content_rating', 50)
    transpose_bool(datanum, 'director_name', 50)
    transpose_bool(datanum, 'actor_1_name', 50)
    transpose_bool(datanum, 'actor_2_name', 50)
    transpose_bool(datanum, 'actor_3_name', 50)
    
    # Suppression de la colonne "vide"
    #del datanames['vide']
    
    # Suprresion de ce qui n'est pas chiffré
    datanum = datanum.drop(['color', 'director_name', 'actor_1_name', 'genres', 'movie_title', 'actor_2_name', 'actor_3_name'], axis=1)
    datanum = datanum.drop(['plot_keywords', 'movie_imdb_link', 'language', 'country', 'content_rating'], axis=1)
    
    # Suprresion de ce qui n'est pas chiffré #2
    datanum = datanum.drop(['num_critic_for_reviews', 'director_facebook_likes', 'actor_3_facebook_likes'], axis=1)
    datanum = datanum.drop(['num_user_for_reviews', 'actor_1_facebook_likes', 'actor_2_facebook_likes'], axis=1)
    datanum = datanum.drop(['aspect_ratio', 'num_voted_users', 'cast_total_facebook_likes'], axis=1)
    
    datanum.fillna(0, inplace=True)
    
    # Coude
    res, dico = affichage_kmeans(datanum, 2, 500, 2)
    
    # C'est moche
    illustration(datanum, range(150, 200, 1))
    
    # PEP 448 et disponible à partir de Python 3.5
    # Mix de deux dictionnaires
    #z = {**res, **save}
    
    # Autre graphique
    plt.bar(range(len(res)), list(res.values()), align='center')
    plt.xticks(range(len(res)), list(res.keys()))
    plt.show()

    # On remets les titres pour y voir plus clair
    datanum['movie_title'] = data['movie_title']    
    
    # Petits tests
    liste_titres = ['Star Wars', 'Expendables', 'American Pie', 'Toy Story', 'Shrek', 'Saw', 'Rocky']
    for titre in liste_titres:
        tester_moteur(datanum, titre)
    
    print(datanum['labels'][datanum['labels'] == 99])
    print(datanum['movie_title'][datanum['labels'] == 99])
    
    histogramme(datanum, 'labels')
    
    datanum.to_csv('C:\\Users\\Toni\\Desktop\\pas_synchro\\p3_test.csv')
    
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
