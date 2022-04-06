


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
удаление корреляций
'''

def delete_corr(df, cut_off = 0.7, exclude = []):
  corr_matrix = df.corr().abs()
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
  f, ax = plt.subplots(figsize=(15, 10))
  plt.title('Все корреляции', fontsize=20)
  sns.heatmap(df.corr(), annot=True)
  
  try:
      f, ax = plt.subplots(figsize=(15, 10))
      plt.title('Высокая корреляция', fontsize=20)
      sns.heatmap(corr_matrix[(corr_matrix>cut_off) & (corr_matrix!=1)].dropna(axis=0, how='all').dropna(axis=1, how='all'), annot=True, linewidths=.5)
  except:
      print ('Нет признаков с высокой коррлеляцией')

  to_drop = [column for column in upper.columns if any(upper[column]>cut_off)]
  to_drop = [column for column in to_drop if column not in exclude]
  print('Удаленные признаки:', to_drop, '\n')
  df2 = df.drop(to_drop, axis = 1)

  f, ax = plt.subplots(figsize=(15, 10))
  plt.title('Финальная корреляция', fontsize = 20)
  sns.heatmap(df2.corr(), annot=True)
  plt.show()
  return df2

delete_corr(df)


'''
Визуализация и анализ
'''

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
df['Day_engaged']=df[['Day_engaged']].apply(LE.fit_transform)

del_cols = ['Age_cut', 'Visits_cut']
df.drop(del_cols, axis = 1, inplace = True)
X = df.drop('Response', axis = 1)
y = df['Response']

scaler = StandardScaler()
scaler.fit(X)
scaled_X = pd.DataFrame(scaler.transform(X),columns= X.columns)
