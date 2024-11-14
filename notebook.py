#coding:utf-8

"""
Bonjour à tous,

Dans ce note book nous allons ...

1 - Importation des données et bibliothèques / data import and libraries import
"""

"""
import pandas as pd
raw_data = pd.read_csv('/home/sectossd/repositories/kaggle/Kgl_Manufacturing_Defects/raw_data.csv',
                       encoding='utf-8',
                       encoding_errors='replace',
                       on_bad_lines='warn',
                       engine='python')
print(raw_data[0:10])
"""

raw_data = open('/home/sectossd/repositories/kaggle/Kgl_Manufacturing_Defects/raw_data.csv',
                encoding='ASCII')
print( raw_data.read(1000) )

# 'rb'
