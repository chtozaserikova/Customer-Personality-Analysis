from sklearn.cluster import DBSCAN 
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.svm import SVC
from xgboost import XGBClassifier


db = DBSCAN(eps=0.726, min_samples=26)
db.fit(X_PCA)
clusters = db.labels_
n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise_ = list(clusters).count(-1)
print("Число кластеров: %d" % n_clusters_)
print("Число шумовых значений: %d" % n_noise_)
print('Количество точек на кластер:')
for i in range(n_clusters_):
    print('Кластер', i, ':', len(clusters[clusters==i]))

    

X_train, X_test, y_train, y_test = train_test_split(X_PCA, y, test_size=0.3, random_state=8)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=8)
print('Длина каждого датасета:')
print('Тренировочный набор:', len(X_train))
print('Валидационный набор:', len(X_val))
print('Тестовый набор:', len(X_test))


sm = SMOTE(random_state=8)
X_bal, y_bal = sm.fit_resample(X_train, y_train)
print('Начальный тренировочный набор')
print('Процент ответа:', y_train.sum()/len(y_train))
print('Сбалансированный тренировочный набор')
print('Процент ответа:', y_bal.sum()/len(y_bal))


lr_params = {'solver': ['liblinear'], 'penalty': ['l1'], 'C': [0.5, 0.25, 0.15, 0.05, 0.01]}
lr_grid = GridSearchCV(LogisticRegression(), lr_params, cv=3, scoring='recall')
lr_grid.fit(X_bal, y_bal)
lr = lr_grid.best_estimator_
print('Лучшие параметры:', lr_grid.best_params_)
lr_preds = lr.predict(X_val)
lr_val_acc = accuracy_score(y_val, lr_preds)
lr_val_rec = recall_score(y_val, lr_preds)
print('Accuracy:', lr_val_acc)
print('Recall:', lr_val_rec)


xgb_params = {'n_estimators': [240, 250, 260], 'max_depth': [15, 16, 17],
             'colsample_bytree': [0.6, 0.7, 0.8, 1.0]}
xgb_grid = GridSearchCV(XGBClassifier(use_label_encoder=False, verbosity=0), xgb_params, cv=3, scoring='recall')
xgb_grid.fit(X_bal, y_bal)
xgb = xgb_grid.best_estimator_
print('Лучшие параметры:', xgb_grid.best_params_)
xgb_preds = xgb.predict(X_val)
xgb_val_acc = accuracy_score(y_val, xgb_preds)
xgb_val_rec = recall_score(y_val, xgb_preds)
print('Accuracy:', xgb_val_acc)
print('Recall:', xgb_val_rec)


knn_params = {'n_neighbors': [5, 7, 9, 11, 15, 20], 'algorithm': ['ball_tree', 'kd_tree', 'brute'],
             'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='recall')
knn_grid.fit(X_bal, y_bal)
knn = knn_grid.best_estimator_
print('Лучшие параметры:', knn_grid.best_params_)
knn_preds = knn.predict(X_val)
knn_val_acc = accuracy_score(y_val, knn_preds)
knn_val_rec = recall_score(y_val, knn_preds)
print('Accuracy:', knn_val_acc)
print('Recall:', knn_val_rec)


clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=160, random_state=10)  
BT_cv_results_acc = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')   
msg = "k-fold Accuracy: %f (%f)" % (BT_cv_results_acc.mean(), BT_cv_results_acc.std())
print(msg)
