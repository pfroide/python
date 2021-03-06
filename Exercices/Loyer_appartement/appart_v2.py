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

# On charge le dataset
house_data_raw = pd.read_csv(Fichier)

# On supprime les prix trop grands
house_data_raw = house_data_raw[house_data_raw['price'] <7000]

# On supprime les lignes qui sont corrompues (et avec des valeurs vides)
house_data = house_data_raw[house_data_raw['surface'] > 0]

# Séparation Train/Testing 20%/80%
xtrain, xtest, ytrain, ytest, ztrain, ztest = train_test_split(house_data['surface'], house_data['price'], house_data['arrondissement'], train_size=0.8)

# Tableau des couleurs pour les plots
couleur=['ro','go','bo','co','ko']

# Valeur initiale qui fait tourner les couleurs de l'array couleur
i=0

# Boucle pour afficher les cinq différents plots suivant les différends arrondissements
for arrondissement in set(ztrain):                      # On utilise set(ztrain) pour avoir une seule fois chaque valeur
    # On fait 5 figures différentes
    plt.subplot(1,5,i+1)

    # Filtre par arrondissement
    xtrain_affiche=xtrain[ztrain==arrondissement]       # [ztrain==value] => condition pour prendre en compte seulement l'arrondissement qui nous intéresse
    ytrain_affiche=ytrain[ztrain==arrondissement]

    # Pour que le plot soit lisible
    plt.grid(True)
    plt.title("Arr : " + str(arrondissement))           # arrondissement correspond à l'arrondissement
    plt.xlabel("Surface")
    plt.ylabel("Loyer")
    plt.xlim(0,150)
    plt.ylim(0,7000)

    # Affichage du nuage de points avec couleur différente à chaque tour
    plt.plot(xtrain_affiche,ytrain_affiche,couleur[i], markersize=3)
    
    # Calcul de la droite optimale
    regr = linear_model.LinearRegression()
    regr.fit(xtrain_affiche.values.reshape(-1, 1), ytrain_affiche)
    regr.predict(xtrain_affiche.values.reshape(-1, 1))  # Prédictions
    
    # Affichage du coeff directeur de la droite
    print('Coefficient : %.2f' % regr.coef_)

    # Affichage de la variances : 1 is perfect prediction
    print('Variance : %.2f' % regr.score(xtrain_affiche.values.reshape(-1, 1), ytrain_affiche))

    # Prédiction pour 100 m2
    prix=regr.predict(100)
    print("Loyer d'un 100m2 dans le " + str(arrondissement) + " arrondissement : " + str(prix) + " euros")
    
    # Affichage de la droite optimale
    plt.plot([0,150],[regr.intercept_,regr.intercept_ + 150*regr.coef_],linestyle='--')
    plt.legend()

    # Incrément
    i=i+1

plt.show()
