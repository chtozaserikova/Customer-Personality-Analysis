


'''
Вычислим сводную статистику по столбцам
'''

col_for_stat = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 
                'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'Age',
                'NumWebVisitsMonth', 'Expenses', 'Total_Accepted', 'Total_Purchases']
num_col = list(df.select_dtypes(['int64', 'float64', 'datetime64[ns]']).columns)
obj_col = list(df.select_dtypes(['object']).columns)

df[col_for_stat].describe().style.background_gradient(cmap='YlOrRd') 


'''
Чистим выбросы (дописать)
'''
def count_outliers(df, col):
  out = []
  q1 = df[col].quantile(0.25, interpolation = 'nearest')
  q3 = df[col].quantile(0.75, interpolation = 'nearest')
  IQR = q3 - q1
  minimum = q1 - 1.5*IQR
  maximum = q3 + 1.5*IQR
  for elem in df[col]:
    if elem < minimum or elem > maximum:
      out.append(df.loc[df[col]==elem].index[0])  
  df.drop(labels = out, axis = 0, inplace = True) 
  
for x in col_for_stat:
  count_outliers(df, x)
  
df.shape




'''
Визуализация и анализ
'''
plt.figure(figsize=(16,9))
ax = sns.heatmap(df[col_for_stat].corr(),annot = True,cmap = 'viridis')
plt.show()

df['Age_cut'] = pd.qcut(df['Age'], q=5)
df.groupby('Age_cut').agg({'Response' : ['mean','count']})

df.groupby('Education').agg({'Response' : ['mean','count']})

df.groupby('Living_Status').agg({'Response' : ['mean','count']})

df['Visits_cut'] = pd.qcut(df['NumWebVisitsMonth'], q=4)
df.groupby('Visits_cut').agg({'Response' : ['mean','count']})



'''
Кодирование и стандартизация
'''

LE=LabelEncoder()
for i in obj_col:
    df[i]=df[[i]].apply(LE.fit_transform)
df['day_engaged']=df[['day_engaged']].apply(LE.fit_transform)

del_cols = ['Age_cut', 'Visits_cut']
df.drop(del_cols, axis = 1, inplace = True)
X = df.drop('Response', axis = 1)
y = df['Response']

scaler = StandardScaler()
scaler.fit(X)
scaled_features = pd.DataFrame(scaler.transform(X),columns= X.columns )
