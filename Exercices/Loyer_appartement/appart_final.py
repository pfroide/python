# Ce fichier est séparé en qautre parties :
#   1 - Importation des librairies nécessaires
#   2 - Affichage unique non-traitée avec les données brutes
#   3 - Affichage après traitement des données. Les deux moyens utilisés pour traiter les données sont :
#       3a - Filtre par arrondissement
#       3b - Filtre suivant le prix et la surface
#   4 - Conclusion avec la solution optimale présentée. Il s'agit de la combinaison de 3a et 3b

# Les valeurs de variances données en commentaires sont une moyenne des valeurs observées. En effet, la
# séparation des training/testing influe de manière aléatoire sur cette valeur.
#############################################################################################################
#   1 - Importation des librairies nécessaires

# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pour la séparation Training/Testing
from sklearn.model_selection import train_test_split

# Régression linéaire
from sklearn import linear_model

# Lieu où se trouve le fichier
Fichier='C:\\Users\\Toni\\Desktop\\WinPython-64bit-3.6.2.0Qt5\\notebooks\\data\\house_data.csv'

##############################################################################################################
#   2 - Affichage unique non-traitée avec les données brutes

# On charge le dataset
house_data_raw = pd.read_csv(Fichier)
    
# On supprime les lignes qui sont corrompues (et avec des valeurs vides)
house_data_raw = house_data_raw[house_data_raw['surface'] > 0]

# Pour que le plot soit lisible (et joli)
plt.grid(True)
plt.xlabel("Surface")
plt.ylabel("Loyer")
plt.xlim(0,200)
plt.ylim(0,7000)
plt.title("2 - Affichage unique non-traitée avec les données brutes")

# Affichage du nuage de points avec couleur différente à chaque tour
plt.plot(house_data_raw['surface'],house_data_raw['price'],'ro', markersize=3)

# Calcul de la droite optimale
regr = linear_model.LinearRegression()
regr.fit(house_data_raw['surface'].values.reshape(-1, 1), house_data_raw['price'])      # On fit le résultat obtenu
regr.predict(house_data_raw['surface'].values.reshape(-1, 1))                           # Prédictions sur le jeu de "test"

# Affichage de la variances : On doit être le plus proche possible de 1
print('Variance : %.2f' % regr.score(house_data_raw['surface'].values.reshape(-1, 1), house_data_raw['price']))

# Affichage de la droite optimale
plt.plot([0,200],[regr.intercept_,regr.intercept_ + 200*regr.coef_],linestyle='--',label=regr.coef_)
plt.legend()

plt.show()
# # # Conclusion :
# On obtient une droite qui n'est pas forcément interprétable à ce stade. Variance = 0,79
##############################################################################################################

##############################################################################################################
#   3 - Affichage après traitement des données.
#       3a - Filtre suivant l'arrondissement

# On recharge le dataset
house_data_raw = pd.read_csv(Fichier)

# On supprime les lignes qui sont corrompues (et avec des valeurs vides)
house_data = house_data_raw[house_data_raw['surface'] > 0]

# Séparation Train/Testing 10%/90%
dtb_xtrain, dtb_xtest, dtb_ytrain, dtb_ytest, dtb_ztrain, dtb_ztest = train_test_split(house_data['surface'], house_data['price'], house_data['arrondissement'], train_size=0.9)

# Tableau des couleurs pour les plots
couleur=['ro','go','bo','co','ko']

# Valeur initiale qui fait tourner les couleurs de l'array couleur
i=0

# Boucle pour afficher les cinq différents plots suivant les différends arrondissements
for arrondissement in set(dtb_ztrain):                              # On utilise set(ztrain) pour avoir une seule fois chaque valeur
    # On fait 5 figures différentes
    plt.subplot(1,5,i+1)

    # Filtre par arrondissement
    dtb_xtrain_filtre=dtb_xtrain[dtb_ztrain==arrondissement]        # [dtb_ztrain==arrondissement] => condition pour prendre en compte seulement l'arrondissement qui nous intéresse
    dtb_ytrain_filtre=dtb_ytrain[dtb_ztrain==arrondissement]
    dtb_xtest_filtre=dtb_xtest[dtb_ztest==arrondissement]
    dtb_ytest_filtre=dtb_ytest[dtb_ztest==arrondissement]
    
    # Pour que le plot soit lisible (et joli)
    plt.grid(True)
    plt.xlabel("Surface")
    plt.ylabel("Loyer")
    plt.xlim(0,200)
    plt.ylim(0,7000)
    if i==0 : plt.title("3a - Filtre suivant l'arrondissement")
    
    # Affichage du nuage de points avec couleur différente à chaque tour
    plt.plot(dtb_xtrain_filtre,dtb_ytrain_filtre,couleur[i], markersize=3)
    
    # Calcul de la droite optimale
    regr = linear_model.LinearRegression()
    regr.fit(dtb_xtrain_filtre.values.reshape(-1, 1), dtb_ytrain_filtre)    # On fit le résultat obtenu
    regr.predict(dtb_xtest_filtre.values.reshape(-1, 1))                    # Prédictions sur le jeu de "test"

    # Affichage de la variances : On doit être le plus proche possible de 1
    print("Arr : " + str(arrondissement))
    print('Variance : %.2f' % regr.score(dtb_xtest_filtre.values.reshape(-1, 1), dtb_ytest_filtre))
    
    # Affichage de la droite optimale
    plt.plot([0,200],[regr.intercept_,regr.intercept_ + 200*regr.coef_],linestyle='--',label=regr.coef_)
    plt.legend()

    # Incrément
    i=i+1

plt.show()
# # # Conclusion :
# On constate ici que les 5 arrondissements ont des prix/m2 relativements différents, surtout le 10ème arr.
# Variance moyenne des 5 graphiques : 0,8
##############################################################################################################

##############################################################################################################
#   3 - Affichage après traitement des données.
#       3b - Filtre suivant le prix et la surface

# On recharge le dataset
house_data_raw = pd.read_csv(Fichier)

# On supprime les prix et les sufraces trop grands
house_data_raw = house_data_raw[house_data_raw['price'] <5000]
house_data_raw = house_data_raw[house_data_raw['surface'] <150]

# On supprime les lignes qui sont corrompues (et avec des valeurs vides)
house_data = house_data_raw[house_data_raw['surface'] > 0]

# Pour que le plot soit lisible (et joli)
plt.grid(True)
plt.xlabel("Surface")
plt.ylabel("Loyer")
plt.xlim(0,200)
plt.ylim(0,7000)
plt.title("3b - Filtre suivant le prix et la surface")

# Affichage du nuage de points avec couleur différente à chaque tour
plt.plot(house_data_raw['surface'],house_data_raw['price'],'ro', markersize=3)

# Calcul de la droite optimale
regr = linear_model.LinearRegression()
regr.fit(house_data_raw['surface'].values.reshape(-1, 1), house_data_raw['price'])      # On fit le résultat obtenu
regr.predict(house_data_raw['surface'].values.reshape(-1, 1))                           # Prédictions sur le jeu de "test"

# Affichage de la variances : On doit être le plus proche possible de 1
print('Variance : %.2f' % regr.score(house_data_raw['surface'].values.reshape(-1, 1), house_data_raw['price']))

# Affichage de la droite optimale
plt.plot([0,200],[regr.intercept_,regr.intercept_ + 200*regr.coef_],linestyle='--',label=regr.coef_)
plt.legend()

plt.show()
# # # Conclusion :
# On constate ici que le fait de ne pas prendre des prix ou surfaces extrèmes améliore la précision du
# résultat. Variance = 0,81, ce qui est mieux que lors de la partie 2.
##############################################################################################################

##############################################################################################################
#   4 - Conclusion avec la solution optimale présentée. Il s'agit de la combinaison de 3a et 3b

# On recharge le dataset
house_data_raw = pd.read_csv(Fichier)

# On supprime les prix et les sufraces trop grands
house_data_raw = house_data_raw[house_data_raw['price'] <5000]
house_data_raw = house_data_raw[house_data_raw['surface'] <150]

# On supprime les lignes qui sont corrompues (et avec des valeurs vides)
house_data = house_data_raw[house_data_raw['surface'] > 0]

# Séparation Train/Testing 10%/90%
dtb_xtrain, dtb_xtest, dtb_ytrain, dtb_ytest, dtb_ztrain, dtb_ztest = train_test_split(house_data['surface'], house_data['price'], house_data['arrondissement'], train_size=0.9)

# Tableau des couleurs pour les plots
couleur=['ro','go','bo','co','ko']

# Valeur initiale qui fait tourner les couleurs de l'array couleur
i=0

# Boucle pour afficher les cinq différents plots suivant les différends arrondissements
for arrondissement in set(dtb_ztrain):                              # On utilise set(ztrain) pour avoir une seule fois chaque valeur
    # On fait 5 figures différentes
    plt.subplot(1,5,i+1)

    # Filtre par arrondissement
    dtb_xtrain_filtre=dtb_xtrain[dtb_ztrain==arrondissement]        # [dtb_ztrain==arrondissement] => condition pour prendre en compte seulement l'arrondissement qui nous intéresse
    dtb_ytrain_filtre=dtb_ytrain[dtb_ztrain==arrondissement]
    dtb_xtest_filtre=dtb_xtest[dtb_ztest==arrondissement]
    dtb_ytest_filtre=dtb_ytest[dtb_ztest==arrondissement]
    
    # Pour que le plot soit lisible (et joli)
    plt.grid(True)
    plt.xlabel("Surface")
    plt.ylabel("Loyer")
    plt.xlim(0,200)
    plt.ylim(0,7000)
    if i==0 : plt.title("4 - Conclusion avec la solution optimale présentée. Il s'agit de la combinaison de 3a et 3b")

    # Affichage du nuage de points avec couleur différente à chaque tour
    plt.plot(dtb_xtrain_filtre,dtb_ytrain_filtre,couleur[i], markersize=3)
    
    # Calcul de la droite optimale
    regr = linear_model.LinearRegression()
    regr.fit(dtb_xtrain_filtre.values.reshape(-1, 1), dtb_ytrain_filtre)    # On fit le résultat obtenu
    regr.predict(dtb_xtest_filtre.values.reshape(-1, 1))                    # Prédictions sur le jeu de "test"

    # Affichage de la variances : On doit être le plus proche possible de 1
    print("Arr : " + str(arrondissement))
    print('Variance : %.2f' % regr.score(dtb_xtest_filtre.values.reshape(-1, 1), dtb_ytest_filtre))
    
    # Affichage de la droite optimale
    plt.plot([0,200],[regr.intercept_,regr.intercept_ + 200*regr.coef_],linestyle='--',label=regr.coef_)
    plt.legend()

    # Incrément
    i=i+1

plt.show()
# # # Conclusion :
# Grace à ces deux améliorations, on arrive à detecter une granularité plus fine dans nos données. En effet,
# les arrondissements 1, 2, 3 et 4 ont des prix/m2 relativement similaires, alors que le 10ème arrondissement
# est moins cher. En gardant le 10ème arrondissement dans la même étude, les résultats sont faussés.
# De même, le fait de ne pas filtrer les valeurs extrèmes rends les prix/m2 des différends arrondissements
# incorrects.
# Variance moyenne des 5 graphiques : 0,86
# On obtient la variance la plus proche de 1 dans ce cas, qui est le meilleur.
#########################################################################################################
