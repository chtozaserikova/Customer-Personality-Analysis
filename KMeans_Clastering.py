from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import plotly.express as px
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer


'''
Уменьшение размерности
'''

variance_ratio = {}
for i in range(3, 13):
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

#видим, что 8 прищнаков объясняют 70% дисперсии, а 90% - 15 признаков

pca = PCA(n_components=8)
pca.fit(scaled_features)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
df_PCA = pca.transform(scaled_features)
df_PCA.shape

'''
Поиск числа кластеров
'''

km = KMeans()
elbow = KElbowVisualizer(estimator = km, k = 10)
elbow.fit(df_PCA)
elbow.show()
#Лучший K по методу локтя равен 4, попробуем метод силуэта.

elbow = KElbowVisualizer(estimator = km, k = 10, metric='silhouette')
elbow.fit(df_PCA)
elbow.show()
#Обычно рекомендуется выбирать номер K со вторым по величине показателем силуэта, поэтому оптимальное число кластеров K = 5

range_n_clusters = list(range(3, 15))
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    preds = clusterer.fit_predict(df_PCA)
    centers = clusterer.cluster_centers_

    score = silhouette_score(df_PCA, preds)
    print("Для числа кластеров равное {}, оценка силуэта = {})".format(n_clusters, score))

# изобразим графически полученный результат
kmeans = KMeans(n_clusters= 5)
label = kmeans.fit_predict(df_PCA)
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(df_PCA[label == i , 0] , df_PCA[label == i , 1] , label = i)
plt.legend()
plt.show()

'''
Подтвердили, что оптимальное число кластеров - 5
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
