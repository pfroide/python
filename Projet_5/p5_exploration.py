# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 19:21:40 2018

@author: Toni
"""

# On importe les librairies dont on aura besoin pour ce tp
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Lieu où se trouve le fichier
_DOSSIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\p5\\'
_DOSSIERTRAVAIL = 'C:\\Users\\Toni\\python\\python\\Projet_5\\images'

def dbscan(data_x, nb_samples):
    """
    Test de l'algorithme dbscan
    """

    data_x = StandardScaler().fit_transform(data_x)

    # Compute DBSCAN
    var_db = DBSCAN(eps=0.5, min_samples=nb_samples).fit(data_x)
    core_samples_mask = np.zeros_like(var_db.labels_, dtype=bool)
    core_samples_mask[var_db.core_sample_indices_] = True
    labels = var_db.labels_
    #print(labels)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    mask = (labels == -1)
    reste = -sum(labels[mask])

    print('N =', nb_samples)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Nombre sans clusters : %d' % reste)
    print("Silhouette Coeff : %0.3f" % metrics.silhouette_score(data_x, labels))

    # Nouvelle colonne avec les conclusions de kmeans
    #X['labels'] = labels

    return labels

def affichage_kmeans(datanum, vmin, vmax, step):
    """
    Test de l'algorithme kmeans et affichage de la méthode du coude.
    """

    # Variables
    listing = dict()
    distortions = []

    # Scale des données obligatoire avant la réduction des dimensions
    #std_scale = MinMaxScaler().fit(datanum)
    #X_scaled = std_scale.transform(datanum)
    std_scale = preprocessing.StandardScaler().fit(datanum)
    x_scaled = std_scale.transform(datanum)

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

        kmeans.fit(x_scaled)
        # Nouvelle colonne avec les conclusions de kmeans
        datanum['labels'] = kmeans.labels_

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        cluster_labels = kmeans.fit_predict(x_scaled)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(x_scaled, cluster_labels)
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
    Affichage d'histogrammes
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

def creation_intervalles_days(df_num, nom_colonne, depart_point, frequence, nb_periods):
    """
    Création des intervalles (pour les jours) où vont être mis les variables
    de durée
    """

    # Convertion de la Série d'entrée en DataFrame
    datatemp = pd.DataFrame(df_num[nom_colonne]).astype('datetime64[ns]')

    # Intervalles
    bins_dt = pd.date_range(depart_point,
                            freq=frequence,
                            periods=nb_periods
                           )
    bins_str = bins_dt.astype(str).values

    # Labels des intervalles
    labels = ['({}, {}]'.format(bins_str[i-1], bins_str[i]) for i in range(1, len(bins_str))]

    # Nom des nouvelles colonnes pour préparer le one-hot encoding
    new_column_name = 'interval_' + nom_colonne

    # df.Date.astype(np.int64)//10**9 - converts datetime values into UNIX epoch
    datatemp[new_column_name] = pd.cut(datatemp[nom_colonne].astype(np.int64)//10**9,
                                       bins=bins_dt.astype(np.int64)//10**9,
                                       labels=labels
                                      )

    # Suppression de la colonne qui existe déjà dans le dataframe final
    del datatemp[nom_colonne]

    # Insertion dans le dataframe qui existait auparavant
    df_num = df_num.join(datatemp, how='right')

    return df_num

def comparaison(datatemp, ligne, colonne):
    """
    Fonction qui déterminer l'intervalle horaire de l'horaire en entrée
    """

    # Cinq comparaisons pour connaitre l'intervalle horaire
    if datatemp.loc[str(ligne), colonne].hour < 8:
        return '[00:00 - 08:00)'
    elif datatemp.loc[str(ligne), colonne].hour < 12:
        return '[08:00 - 12:00)'
    elif datatemp.loc[str(ligne), colonne].hour < 16:
        return '[12:00 - 16:00)'
    elif datatemp.loc[str(ligne), colonne].hour < 20:
        return '[16:00 - 20:00)'
    #Sinon
    return '[20:00 - 00:00)'

def creation_intervalles_heures(df_num, nom_colonne):
    """
    Appel de la fonction pour désigner l'intervalle horaire
    """

    # Convertion de la Série d'entrée en DataFrame
    datatemp = pd.DataFrame(df_num[nom_colonne])

    # Nom des nouvelles colonnes pour préparer le one-hot encoding
    new_column_name = 'interval_' + nom_colonne

    # Pour tous les index, on recherche les données
    for ligne in datatemp.index:
        datatemp.loc[str(ligne), new_column_name] = comparaison(datatemp, ligne, nom_colonne)

    # Suppression de la colonne qui existe déjà dans le dataframe final
    del datatemp[nom_colonne]

    # Insertion dans le dataframe qui existait auparavant
    df_num = df_num.join(datatemp, how='right')

    return df_num

def calcul_sommes(data, df_num):
    """
    Fonction qui va gérer toutes les informations retirées graçe aux dates
    """

    # Colonnes que l'on va créer dans le dataframe vide
    colonnes = ['valeur_facture_1',
                'valeur_facture_2',
                'valeur_last_facture'
               ]

    # Création d'un dataframe vide
    datatemp = pd.DataFrame(columns=colonnes)

    # Création de la liste des valeurs des factures
    aggregations = {'TotalAmount':'sum'}
    liste_amounts = data.groupby('InvoiceNo').agg(aggregations)

    # Pour tous les index, on recherche les données
    for ligne in df_num.index:
        # Récupération de toutes les dates d'achats
        liste_factures = pd.Series(data['InvoiceNo'][data['CustomerID'] == float(ligne)].unique())

        # Récupération de la valeur de la première facture, case 0
        premiere_facture = liste_amounts.loc[liste_factures[0]][0]

        # Et on rajoute ça dans le dataframe créé pour ça
        datatemp.loc[str(ligne), colonnes[0]] = premiere_facture

        # Récupération de la valeur d'autres factures
        if len(liste_factures) > 1:
            # Facture n°2
            deuxieme_facture = liste_amounts.loc[liste_factures[1]][0]

            # Dernière facture
            derniere_facture = liste_amounts.loc[liste_factures[len(liste_factures)-1]][0]

            # Et on rajoute ça dans le dataframe créé pour ça
            datatemp.loc[str(ligne), colonnes[1]] = deuxieme_facture
            datatemp.loc[str(ligne), colonnes[2]] = derniere_facture

    # Insertion dans le dataframe qui existait auparavant
    df_num = df_num.join(datatemp, how='right')

    return df_num

def calcul_dates(data, df_num):
    """
    Fonction qui va gérer toutes les informations retirées graçe aux dates
        * nb_jour_2_achats
        * moyenne_horaire
        * jour_achat_n1
        * heure_achat_n1
        * jour_dernier_achat
        * heure_dernier_achat
    """

    # Colonnes que l'on va créer dans le dataframe vide
    colonnes = ['ecart_moy_2_achats',
                'moyenne_horaire',
                'jour_achat_n1',
                'heure_achat_n1',
                'jour_dernier_achat',
                'heure_dernier_achat',
                'ecart_min_2_achats',
                'ecart_max_2_achats'
               ]

    # Création d'un dataframe vide
    datatemp = pd.DataFrame(columns=colonnes)

    # Pour tous les index, on recherche les données
    for ligne in df_num.index:
        # Récupération de toutes les dates d'achats
        dates = pd.Series(data['InvoiceDate'][data['CustomerID'] == float(ligne)].unique())

        # Initialisation des variables au bon type
        somme = datetime.timedelta(0)
        ecart_min = datetime.timedelta(500)
        ecart_max = datetime.timedelta(0)
        res = []

        # Pour toutes les dates, on calcule la différence entre 2 dates
        for i, j in enumerate(dates):
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

        # Vérification qu'il y a eu plus qu'un achat, sinon on enlève la valeur initiale
        if ecart_min == datetime.timedelta(500):
            ecart_min = datetime.timedelta(0)

        # On fait la moyenne à la fin
        moyenne = somme / len(dates)

        # Formatage des heures et minutes pour l'heure moyenne d'achat
        var_heures = int((sum(res)/len(res))/60)
        var_minutes = int(sum(res)/len(res) - var_heures*60)

        # Et on rajoute ça dans le dataframe créé pour ça
        datatemp.loc[str(ligne), colonnes[0]] = moyenne.days
        datatemp.loc[str(ligne), colonnes[1]] = datetime.time(var_heures,
                                                              var_minutes)
        datatemp.loc[str(ligne), colonnes[2]] = dates[0].date()
        datatemp.loc[str(ligne), colonnes[3]] = datetime.time(dates[0].hour,
                                                              dates[0].minute)
        datatemp.loc[str(ligne), colonnes[4]] = dates[len(dates)-1].date()
        datatemp.loc[str(ligne), colonnes[5]] = datetime.time(dates[len(dates)-1].hour,
                                                              dates[len(dates)-1].minute)
        datatemp.loc[str(ligne), colonnes[6]] = ecart_min.days
        datatemp.loc[str(ligne), colonnes[7]] = ecart_max.days

    # Insertion dans le dataframe qui existait auparavant
    df_num = df_num.join(datatemp, how='right')

    return df_num

def main():
    """
    Fonction principale
    """

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

    # Premiers calculs qui sont fait via l'appel d'un dictionnaire
    aggregations = {'TotalAmount':'sum',
                    'Quantity':'sum',
                    'InvoiceNo': ['count', pd.Series.nunique]
                   }

    df_num = data.groupby('CustomerID').agg(aggregations)

    # Renommage des colonnes du résultat de l'aggrégation
    df_num.columns = ['1', '2', '3', '4']
    df_num = df_num.rename(index=str, columns={"1": "somme_total",
                                               "2": "nb_article_total",
                                               "3": "nb_categorie_article_total",
                                               "4": "nb_factures"})

    # Calculs de moyennes
    df_num['nb_article_moyen_par_facture'] = df_num['nb_article_total']/df_num['nb_factures']
    df_num['somme_moyenne_par_facture'] = df_num['somme_total']/df_num['nb_factures']
    df_num['nb_categorie_article_par_facture'] = df_num['nb_categorie_article_total']/df_num['nb_factures']
    df_num['somme_moyenne_par_article'] = df_num['somme_total']/df_num['nb_article_total']

    # Appel de la fonction qui va gérer tout à partir des dates
    df_num = calcul_dates(data, df_num)

    # Formatage en int
    df_num['ecart_moy_2_achats'] = df_num['ecart_moy_2_achats'].astype(int)
    df_num['ecart_min_2_achats'] = df_num['ecart_min_2_achats'].astype(int)
    df_num['ecart_max_2_achats'] = df_num['ecart_max_2_achats'].astype(int)

    # Appel de la fonction qui gérer les différentes factures
    df_num = calcul_sommes(data, df_num)

    # Formatage en intervalle des dates
    nom_colonne = ['jour_achat_n1', 'jour_dernier_achat']

    # Création des intervalles journaliers pour les nom_colonne ci-dessus
    for i in range(0, len(nom_colonne)):
        df_num = creation_intervalles_days(df_num,
                                           nom_colonne[i],
                                           '2010-11-01',
                                           '1M',
                                           14
                                          )
        del df_num[nom_colonne[i]]

    # Formatage en intervalle des heures
    nom_colonne = ['moyenne_horaire', 'heure_achat_n1', 'heure_dernier_achat']

    # Création des intervalles horaires pour les nom_colonne ci-dessus
    for i in range(0, len(nom_colonne)):
        df_num = creation_intervalles_heures(df_num, nom_colonne[i])
        del df_num[nom_colonne[i]]

    # Pour vérifier
    data[data['CustomerID'] == 12347]
    data['InvoiceNo'][data['CustomerID'] == 12347].count()

    # One-Hot encoding
    liste_criteres = ['interval_moyenne_horaire',
                      'interval_jour_achat_n1',
                      'interval_heure_achat_n1',
                      'interval_jour_dernier_achat',
                      'interval_heure_dernier_achat'
                     ]
    df_num = pd.get_dummies(data=df_num, columns=liste_criteres)

    # Méthode du KMeans (coude)
    df_num.fillna(0, inplace=True)
    res, dico = affichage_kmeans(df_num, 3, 40, 1)

    # Répartition des labels
    histogramme(df_num, 'labels')

    # Tester dbscan
    for i in range(1, 25):
        liste = dbscan(df_num, i)

#    res7 = df_num.groupby('labels').size().reset_index(name='nb')
#    res7 = df_num.groupby(['labels', 'nb_factures']).size().reset_index(name='nb')

#    table = pd.pivot_table(res8,
#                           values=["nb_factures", "nb_article_total", "somme_total"],
#                           index="CustomerID")

#    # Number of transactions with anonymous customers
#    print(data[data['CustomerID'].isnull()]['InvoiceNo'].unique())

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
