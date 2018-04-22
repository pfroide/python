# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 19:21:40 2018

@author: Toni
"""

# On importe les librairies dont on aura besoin pour ce tp
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm as cm

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.metrics import silhouette_samples, silhouette_score
import json

# Lieu où se trouve le fichier
_DOSSIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\p5\\'
_DOSSIERTRAVAIL = 'C:\\Users\\Toni\\python\\python\\Projet_5\\images'

def dbscan(X, sp):
    """
    TBD
    """

    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(X)

    # Compute DBSCAN
    db = DBSCAN(eps=0.5, min_samples=sp).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    #print(labels)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    mask = (labels == -1)
    reste = -sum(labels[mask])

    print('N =', sp)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Nombre sans clusters : %d' % reste)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

    # Nouvelle colonne avec les conclusions de kmeans
    #X['labels'] = labels

    return labels

def affichage_kmeans(datanum, vmin, vmax, step):
    """
    TBD
    """

    listing = dict()
    distortions = []

    from sklearn.preprocessing import MinMaxScaler

    #
    std_scale = MinMaxScaler().fit(datanum)
    X_scaled = std_scale.transform(datanum)

    # Scale des données obligatoire avant la réduction des dimensions
    #std_scale = preprocessing.StandardScaler().fit(datanum)
    #X_scaled = std_scale.transform(datanum)

    # Réduction t-Sne
    #print("Computing t-SNE embedding")
    #tsne = manifold.TSNE(n_components=2, perplexity=50, n_iter=500)

    cluster_range = range(vmin, vmax+1, step)

    for i in cluster_range:

        # On fait i clusters avec les données scalées.
        kmeans = KMeans(n_clusters=i,
                        n_init=50,
                        max_iter=500,
                        random_state=10)

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
              "The average silhouette_score is :", round(silhouette_avg, 3))

        distortions.append(kmeans.inertia_)
        listing[i] = silhouette_avg

    # Données du graphique du coude
    clusters_df = pd.DataFrame({"num_clusters":cluster_range, "cluster_errors": distortions})
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

def histogramme(data, colon):
    """
    TBD
    """

    fichier_save = _DOSSIERTRAVAIL + '\\' + 'histogram_' + colon

    #steps = (max(data[colon])-min(data[colon]))/100
    #bin_values = np.arange(start=min(data[colon]), stop=max(data[colon]), step=steps)
    plt.figure(figsize=(10, 6))
    plt.xlabel('Valeurs')
    plt.ylabel('Décompte')
    titre = 'Histogramme ' + colon
    plt.title(titre)

    # Affichage sans les valeurs NaN
    plt.hist(data[colon][np.isfinite(data[colon])], bins=100)
    # Avec :
    # plt.hist(data[colon], bins=bin_values)
    plt.savefig(fichier_save, dpi=100)

def calcul_dates(data, res6):
    """
    Fonction qui va gérer toutes les informations retirées graçe aux dates
        * nb_jour_entre_2_achats
        * moyenne_horaire
        * jour_achat_n1
        * heure_achat_n1
        * jour_dernier_achat
        * heure_dernier_achat
    """

    # Colonnes que l'on va créer dans le dataframe vide
    colonnes = ['ecart_moyen_entre_2_achats',
                'moyenne_horaire',
                'jour_achat_n1',
                'heure_achat_n1',
                'jour_dernier_achat',
                'heure_dernier_achat',
                'ecart_min_entre_2_achats',
                'ecart_max_entre_2_achats'
               ]

    # Création d'un dataframe vide
    newDF = pd.DataFrame(columns=colonnes)

    # Pour tous les index, on recherche les données
    for ligne in res6.index:
        # Récupération de toutes les dates d'achats
        dates = pd.Series(data['InvoiceDate'][data['CustomerID'] == float(ligne)].unique())

        # Initialisation des variables au bon type
        somme = datetime.timedelta(0)
        ecart_min = datetime.timedelta(500)
        ecart_max = datetime.timedelta(0)
        res = []

        # Pour toutes les dates, on calcule la différence entre 2 dates
        for i, p in enumerate(dates):
            # Pour calculer l'heure moyenne d'achat
            res.append(dates[i].hour*60 + dates[i].minute)

            # Pour calculer le nombre de jours entre deux achats
            if i != 0:
                somme = somme + (dates[i] - dates[i-1])

                #Comparaison pour les écarts min et max
                if ecart_min > (dates[i] - dates[i-1]):
                    ecart_min = dates[i] - dates[i-1]
                if ecart_max < dates[i] - dates[i-1]:
                    ecart_max = dates[i] - dates[i-1]

        # On fait la moyenne à la fin
        moyenne = somme / len(dates)

        # Formatage des heures et minutes pour l'heure moyenne d'achat
        var_heures = int((sum(res)/len(res))/60)
        var_minutes = int(sum(res)/len(res) - var_heures*60)

        # Et on rajoute ça dans le dataframe créé pour ça
        newDF.loc[str(ligne), colonnes[0]] = moyenne.days
        newDF.loc[str(ligne), colonnes[1]] = datetime.time(var_heures,
                                                           var_minutes)
        newDF.loc[str(ligne), colonnes[2]] = dates[0].date()
        newDF.loc[str(ligne), colonnes[3]] = datetime.time(dates[0].hour,
                                                           dates[0].minute)
        newDF.loc[str(ligne), colonnes[4]] = dates[len(dates)-1].date()
        newDF.loc[str(ligne), colonnes[5]] = datetime.time(dates[len(dates)-1].hour,
                                                           dates[len(dates)-1].minute)
        newDF.loc[str(ligne), colonnes[6]] = ecart_min.days
        newDF.loc[str(ligne), colonnes[7]] = ecart_max.days

    # Insertion dans le dataframe qui existait auparavant
    res6 = res6.join(newDF, how='right')

    return res6

def main():

    # Récupération du dataset
    fichier = 'Online Retail.xlsx'
    data = pd.read_excel(_DOSSIER + fichier, error_bad_lines=False)

    # Données manquantes
    fichier_save = _DOSSIERTRAVAIL + '\\' + 'missing_data.csv'
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

    # Number of transactions with anonymous customers
    print(data[data['CustomerID'].isnull()]['InvoiceNo'].unique())

    data = data[data['StockCode'] != "AMAZONFEE"]
    data = data[data['Quantity'] > 0]
    data = data[pd.notnull(data['CustomerID'])]

    # Add extra fields
    data['TotalAmount'] = data['Quantity'] * data['UnitPrice']
    data['InvoiceYear'] = data['InvoiceDate'].dt.year
    data['InvoiceMonth'] = data['InvoiceDate'].dt.month
    data['InvoiceDay'] = data['InvoiceDate'].dt.day
    data['InvoiceYearMonth'] = data['InvoiceYear'].map(str) + "-" + data['InvoiceMonth'].map(str)
    data['InvoiceMonthDay'] = data['InvoiceMonth'].map(str) + "-" + data['InvoiceDay'].map(str)

    aggregations = {'TotalAmount':'sum',
                    'Quantity':'sum',
                    'InvoiceNo': ['count', pd.Series.nunique]
                    #'InvoiceNo': lambda x: x.unique().count()
                   }

    res6 = data.groupby('CustomerID').agg(aggregations)

    res6.columns = ['1', '2', '3', '4']
    res6 = res6.rename(index=str, columns={"1": "somme_total",
                                           "2": "nb_article_total",
                                           "3": "nb_categorie_article_total",
                                           "4": "nombre_factures"})

    res6['nb_article_moyen_par_facture'] = res6['nb_article_total']/res6['nombre_factures']
    res6['somme_moyenne_par_facture'] = res6['somme_total']/res6['nombre_factures']
    res6['nb_categorie_article_par_facture'] = res6['nb_categorie_article_total']/res6['nombre_factures']
    res6['somme_moyenne_par_article'] = res6['somme_total']/res6['nb_article_total']

    # Appel de la fonction qui va gérer tout à partir des dates
    res6 = calcul_dates(data, res6)

    # Formattage en int
    res6['ecart_moyen_entre_2_achats'] = res6['ecart_moyen_entre_2_achats'].astype(int)
    res6['ecart_min_entre_2_achats'] = res6['ecart_min_entre_2_achats'].astype(int)
    res6['ecart_max_entre_2_achats'] = res6['ecart_max_entre_2_achats'].astype(int)

    # Pour vérifier
    data[data['CustomerID'] == 12347]
    data['InvoiceNo'][data['CustomerID'] == 12347].count()

    # Méthode du KMeans (coude)
    #res6.fillna(0, inplace=True)
    res, dico = affichage_kmeans(res6, 3, 30, 1)

    # Répartition des labels
    histogramme(res6, 'labels')

    # Tester dbscan
    for i in range(1, 15):
        liste = dbscan(res6, i)

    res7 = res6.groupby('labels').size().reset_index(name='nb')
    res7 = res6.groupby(['labels', 'nombre_factures']).size().reset_index(name='nb')

    table = pd.pivot_table(res8,
                           values=["nombre_factures", "nb_article_total", "somme_total"],
                           index="CustomerID")

#    # Total number of transactions
#    len(data['InvoiceNo'].unique())
#
#    # Number of transactions with anonymous customers
#    len(data[data['CustomerID'].isnull()]['InvoiceNo'].unique())
#
#    # Total numbers of customers - +1 for null users
#    len(data['CustomerID'].unique())
#
#    # Get top ranked ranked customers based on the total amount
#    customers_amounts = data.groupby('CustomerID')['TotalAmount'].agg(np.sum).sort_values(ascending=False)
#    customers_amounts.head(20)
#
#    customers_amounts.head(20).plot.bar()
#
#    # Explore by month
#    gp_month = data.sort_values('InvoiceDate').groupby(['InvoiceYear', 'InvoiceMonth'])
#
#    # Month number of invoices
#    gp_month_invoices = gp_month['InvoiceNo'].unique().agg(np.size)
#    gp_month_invoices
#    gp_month_invoices.plot.bar()
#
#    res = data.groupby(['CustomerID', 'InvoiceNo']).size().reset_index(name='nb_item_par_facture')
#    res2 = res.groupby(['CustomerID']).size().reset_index(name='nb_factures')
#    del res['InvoiceNo']
#
#    res_df = data.groupby(['CustomerID', 'Quantity']).size().reset_index(name='nb')
#    res3 = res_df.groupby(['CustomerID']).sum().reset_index()
#	res_df = data.groupby(['CustomerID', 'Date']).size().reset_index(name='nb')
#	dates = data['InvoiceDate'][data['CustomerID'] == 12347].unique().astype('datetime64[D]')
#	dates = pd.Series(data['InvoiceDate'][data['CustomerID'] == 12347].unique())

#    # Define the aggregation procedure outside of the groupby operation
#    aggregations = {'TotalAmount':['sum', min, max, 'mean', 'mad', 'median', 'std', 'sem', 'skew'],
#                    'Quantity':['sum', min, max, 'mean', 'mad', 'median', 'std', 'sem', 'skew'],
#                    'InvoiceNo':'count',
#                    #'date': lambda x: max(x) - 1
#                   }

#def calcul_dates(data, res6):
#    """
#    Fonction qui va gérer toutes les informations retirées graçe aux dates
#        * nb_jour_entre_2_achats
#        * moyenne_horaire
#        * jour_achat_n1
#        * heure_achat_n1
#        * jour_dernier_achat
#        * heure_dernier_achat
#    """
#
#    # Création d'un dataframe avec une colonne vide
#    newDF1 = pd.DataFrame(res6.index, columns=['nb_jour_entre_2_achats'])
#    newDF2 = pd.DataFrame(res6.index, columns=['moyenne_horaire'])
#    newDF3 = pd.DataFrame(res6.index, columns=['jour_achat_n1'])
#    newDF4 = pd.DataFrame(res6.index, columns=['heure_achat_n1'])
#    newDF5 = pd.DataFrame(res6.index, columns=['jour_dernier_achat'])
#    newDF6 = pd.DataFrame(res6.index, columns=['heure_dernier_achat'])
#
#    # Pour tous les index, on recherche les données
#    for ligne in res6.index:
#        # Récupération de toutes les dates d'achats
#        dates = pd.Series(data['InvoiceDate'][data['CustomerID'] == float(ligne)].unique())
#
#        # Initialisation de la variable "somme" au bon type
#        somme = datetime.timedelta(0)
#        res = []
#
#        # Pour toutes les dates, on calcule la différence entre 2 dates
#        for i, p in enumerate(dates):
#            # Pour calculer l'heure moyenne d'achat
#            res.append(dates[i].hour*60 + dates[i].minute)
#
#            # Pour calculer le nombre de jours entre deux achats
#            if i != 0:
#                somme = somme + (dates[i] - dates[i-1])
#
#        # On fait la moyenne à la fin
#        moyenne = somme / len(dates)
#
#        # Formatage des heures et minutes pour l'heure moyenne d'achat
#        var_heures = int((sum(res)/len(res))/60)
#        var_minutes = int(sum(res)/len(res) - var_heures*60)
#
#        # Et on rajoute ça dans le dataframe créé pour ça
#        newDF1.loc[str(ligne)] = moyenne.days
#        newDF2.loc[str(ligne)] = datetime.time(var_heures, var_minutes)
#        newDF3.loc[str(ligne)] = dates[0].date()
#        newDF4.loc[str(ligne)] = datetime.time(dates[0].hour, dates[0].minute)
#        newDF5.loc[str(ligne)] = dates[len(dates)-1].date()
#        newDF6.loc[str(ligne)] = datetime.time(dates[len(dates)-1].hour, dates[len(dates)-1].minute)
#
#    # Insertion dans le dataframe qui existait auparavant
#    res6 = res6.join(newDF1, how='right')
#    res6 = res6.join(newDF2, how='right')
#    res6 = res6.join(newDF3, how='right')
#    res6 = res6.join(newDF4, how='right')
#    res6 = res6.join(newDF5, how='right')
#    res6 = res6.join(newDF6, how='right')
#
#    return res6
