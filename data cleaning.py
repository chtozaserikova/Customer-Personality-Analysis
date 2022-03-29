import numpy as np 
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)

df = pd.read_csv('/marketing_campaign.csv', sep="\t")

df.isnull().sum() * 100 / len(df)

print("Размер датафрейма:", df.shape)

print("Названия столбцов: \n", df.columns)

print('Процент нулевых значений: ', df.isnull().sum() * 100 / len(df))

df = df.dropna()
df = df.drop_duplicates()
print("Количество строк после удаления пропущенных значений и дубликатов:", len(df))

df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
print('Дата регистрации первого клиента: ', df.Dt_Customer.min())
print('Дата регистрации последнего клиента: ', df.Dt_Customer.max())

df["Age"] = 2022-df["Year_Birth"]

df['Marital_Status'].unique()

df["Living_Status"]=df["Marital_Status"].replace({"Married":"Family", "Together":"Family", 
                                                    "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", 
                                                    "Divorced":"Alone", "Single":"Alone"})

df['Education'].unique()

df["Education"]=df["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", 
                                         "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

df['Kids'] = df['Kidhome'] + df['Teenhome']
df['Expenses'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
df['Total_Accepted'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5']
df['Total_Purchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'] + df['NumDealsPurchases']

col_to_drop = ['Year_Birth', 'Marital_Status', 'ID', 'Kidhome', 'Teenhome', 'Dt_Customer']
df.drop(col_to_drop, inplace = True, axis=1)



'''
Найдем неинформативные признаки, у которых более 95% строк содержат одно и то же значение
'''
num_rows = len(df.index)
low_information_cols = [] 
for col in df.columns:
    cnts = df[col].value_counts(dropna=False)
    top_pct = (cnts/num_rows).iloc[0]
    if top_pct > 0.95:
        low_information_cols.append(col)
        print('{0}: {1:.5f}%'.format(col, top_pct*100))
        print(cnts)
        print()

df['Z_CostContact'].unique()
df['Z_Revenue'].unique()

#Данные признаки имеют единственное значение, соответственно не важны при построении модели
df = df.drop(['Z_CostContact','Z_Revenue'],axis = 1)
