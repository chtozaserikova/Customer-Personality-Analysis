from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import plotly.express as px
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
import plotly.graph_objects as go 


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
'''

kmeans = KMeans(n_clusters=5, random_state=10)
predictions = kmeans.fit_predict(df_PCA)
df["Clusters"] = predictions + 1


labels = ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4" ]
cluster0_num = df[df["Clusters"]==1].shape[0]
cluster1_num = df[df["Clusters"]==2].shape[0]
cluster2_num = df[df["Clusters"]==3].shape[0]
cluster3_num = df[df["Clusters"]==4].shape[0]
cluster4_num = df[df["Clusters"]==5].shape[0]
values = [cluster0_num, cluster1_num, cluster2_num, cluster3_num, cluster4_num]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.6, title="Clusters")])
fig.show()
