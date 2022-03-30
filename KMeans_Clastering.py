from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import plotly.express as px
from sklearn.decomposition import PCA


'''
Уменьшение размерности
'''

variance_ratio = {}
for i in range(1, len(scaled_features.columns)+1):
    pca = PCA(n_components=i)
    pca.fit(scaled_features)
    variance_ratio[f'n_{i}'] = pca.explained_variance_ratio_.sum()
variance_ratio 


plt.figure(figsize = (12, 5))
plt.plot([key for key in variance_ratio.keys()], [val for val in variance_ratio.values()])
plt.axhline(0.7, color = 'red', ls = '--', lw = 1)
plt.axhline(0.9, color = 'red', ls = '--', lw = 1.5)
plt.title("Variance Ratio")
plt.show()

#видим, что при доли от общей дисперсии равной 70% нужно взять 8 признаков, при 90% - 15 признаков


'''
собственное значение для рассчитанной ковариационной матрицы
'''
eigen_value = np.sort(pca.explained_variance_)[::-1]
plt.figure(figsize=(10, 10))
plt.plot([key for key in variance_ratio.keys()], eigen_value)
plt.ylim(0, 10, 1)
plt.axhline(1, color = 'red', ls = '--')
plt.title('Elbow Point')
plt.show()

print(f'число собственных значений больше 1 (10%): {len(eigen_value[eigen_value > 1])}')

#получили 7 признаков


'''
Поиск числа кластеров
'''

range_n_clusters = list(range(2, 7))
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    preds = clusterer.fit_predict(X)
    centers = clusterer.cluster_centers_

    score = silhouette_score(X, preds)
    print("Для числа кластеров равное {}, оценка силуэта = {})".format(n_clusters, score))
    
'''
Оптимальное число кластеров - 3
Перейдем к созданию модели
'''
    
gmm = GaussianMixture(n_components = 4, covariance_type = 'spherical', max_iter = 3000, random_state = 228).fit(X)
labels = gmm.predict(X)
X['Cluster'] = labels
re_clust = {
    0: 'Ordinary client',
    1: 'Elite client',
    2: 'Good client',
    3: 'Potential good client'}
X['Cluster'] = X['Cluster'].map(re_clust)


fig = px.pie(X['Cluster'].value_counts().reset_index(), values = 'Cluster', names = 'index', width = 700, height = 700)
fig.update_traces(textposition = 'inside', textinfo = 'percent + label', hole = 0.8, 
                  marker = dict(colors = ['#dd4124','#009473', '#336b87', '#b4b4b4'], line = dict(color = 'white', width = 2)),
                  hovertemplate = 'Clients: %{value}')
fig.update_layout(annotations = [dict(text = 'Number of clients <br>by cluster', 
                                      x = 0.5, y = 0.5, font_size = 28, showarrow = False, 
                                      font_family = 'monospace',
                                      font_color = 'black')],
                  showlegend = False)                  
fig.show()
