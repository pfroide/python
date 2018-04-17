# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 19:21:40 2018

@author: Toni
"""

# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm as cm

# Lieu où se trouve le fichier
_DOSSIER = 'C:\\Users\\Toni\\Desktop\\pas_synchro\\p5\\'
_DOSSIERTRAVAIL = 'C:\\Users\\Toni\\python\\python\\Projet_5\\images'

def main():
    
    # Récupération du dataset
    fichier = 'Online Retail.xlsx'
    data = pd.read_excel(_DOSSIER + fichier, error_bad_lines=False)

    # Données manquantes
    print("Données manquantes")
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
    
    #
    #data['CustomerID'] = data['CustomerID'].astype('float', copy=False)
    mask = data[data["CustomerID"].isnull()]
     
#chercherà faire correspondre les montants négatifs avec les positifs de même valeur
    
    # Add extra fields 
    data['TotalAmount'] = data['Quantity'] * data['UnitPrice']
    data['InvoiceYear'] = data['InvoiceDate'].dt.year
    data['InvoiceMonth'] = data['InvoiceDate'].dt.month
    data['InvoiceYearMonth'] = data['InvoiceYear'].map(str) + "-" + data['InvoiceMonth'].map(str)

    # Total number of transactions
    len(data['InvoiceNo'].unique())
    
    # Number of transactions with anonymous customers 
    len(data[data['CustomerID'].isnull()]['InvoiceNo'].unique())
    
    # Total numbers of customers - +1 for null users
    len(data['CustomerID'].unique())
    
    # Get top ranked ranked customers based on the total amount
    customers_amounts = data.groupby('CustomerID')['TotalAmount'].agg(np.sum).sort_values(ascending=False)
    customers_amounts.head(20)
    
    customers_amounts.head(20).plot.bar()
    
    # Explore by month
    gp_month = data.sort_values('InvoiceDate').groupby(['InvoiceYear', 'InvoiceMonth'])
    
    # Month number of invoices
    gp_month_invoices = gp_month['InvoiceNo'].unique().agg(np.size)
    gp_month_invoices
    gp_month_invoices.plot.bar()

    res = data.groupby(['CustomerID', 'InvoiceNo']).size().reset_index(name='nb_item_par_facture')
    res2 = res.groupby(['CustomerID']).size().reset_index(name='nb_factures')
    del res['InvoiceNo']
    
    res_df = data.groupby(['CustomerID', 'Quantity']).size().reset_index(name='nb')
    res3 = res_df.groupby(['CustomerID']).sum().reset_index()
    
    # Define the aggregation procedure outside of the groupby operation
    aggregations = {'TotalAmount':['sum', min, max, 'mean', 'mad', 'median', 'std', 'sem', 'skew'],
                    'Quantity':['sum', min, max, 'mean', 'mad', 'median', 'std', 'sem', 'skew'],
                    'InvoiceNo':'count',
                    #'date': lambda x: max(x) - 1
                   }
    
    res6 = data.groupby('CustomerID').agg(aggregations)