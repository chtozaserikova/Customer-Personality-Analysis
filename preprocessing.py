


'''
Вычислим сводную статистику по столбцам
'''

num_col = list(df.select_dtypes(['int64', 'float64', 'datetime64[ns]']).columns)
obj_col = list(df.select_dtypes(['object']).columns)

df.loc[:, df.columns != 'ID'].describe().style.background_gradient(cmap='YlOrRd')  

#разберись с глобальными переменными и перепиши адекватно 
counts = []
def count_outliers(data,col):
        q1 = data[col].quantile(0.25, interpolation = 'nearest')
        q3 = data[col].quantile(0.75, interpolation = 'nearest')
        IQR = q3 - q1
        
        minimum = q1 - 1.5*IQR
        maximum = q3 + 1.5*IQR
        if data[col].min() < minimum or data[col].max() > maximum:
            print("Есть выбросы в признаке", i)
            x = data[data[col] < minimum][col].size
            y = data[data[col] > maximum][col].size
            counts.append(i)
            print('Число выбросов: ', x+y, '\n')
for i in df[num_col]:
    count_outliers(df,i)


'''
Визуализация и анализ
'''
plt.figure(figsize=(16,9))
ax = sns.heatmap(df.corr(),annot = True,cmap = 'viridis')
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
 
df['Dt_Customer']=df[['Dt_Customer']].apply(LE.fit_transform)

del_cols = ['Age_cut', 'Visits_cut', 'Response', 'ID']
X = df.drop(del_cols, axis = 1)
y = df['Response']

scaler = StandardScaler()
scaler.fit(X)
scaled_features = pd.DataFrame(scaler.transform(X),columns= X.columns )
