# -*- coding: utf-8 -*-
"""pipeline.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UraMlD1vmkOHINLPHxdz7aEe6xitXr5U

Импорт библиотек для анализа данных и обучения модели
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

"""Загрузка данных в среду Jupyter Notebook"""

!gdown 1bhH6DR0qxrhcA4OmUUhybHFJ_I3vDV3P

df = pd.read_csv('ebw_data.csv')

df.head()

df

"""Обзор данных"""

df.shape

df.describe()

df.info()

for column in df:
  print(f'{column}: количество уникальных значений: {df[column].nunique()}, {df[column].dtype}')

features = df.iloc[:,:-2]

features[features.duplicated()]

features.sort_values(['IW', 'IF', 'VW', 'FP'])

df = df.drop_duplicates()

df.shape

sns.pairplot(df);

df_group = df.groupby([df.IW, df.IF, df.VW, df.FP], as_index=False).mean()

sns.pairplot(df_group);

df.corr()

cm = np.corrcoef(df.values.T)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
            annot_kws={'size': 12}, yticklabels=df.columns, xticklabels=df.columns);

"""Добавим характеристику ускоряющее напряжение и мощность луча по формуле Q = IW * U (сила тока луча на ускоряющее напряжение) """

df['U'] = 20
df['Q'] = df.IW * df.U

df.head()

X = df.drop(['Depth', 'Width'], axis=1).values

X

y = df[['Depth', 'Width']].values

y

"""Нормализуем данные"""

scaler = StandardScaler()

scaler.fit(X)

X_scal = scaler.transform(X)

X_scal[:5,:]

"""Разбиваем данные для обучения и для тестирования"""

X_train, X_test, y_train, y_test = train_test_split(X_scal, y, train_size=0.8, random_state=42)

"""Предварительные результаты моделей, далее выберем модель с лучшими результатами и подберем параметры"""

def test_model(estimator):
  print(estimator,'\n')
  estimator.fit(X_train, y_train)
  y_pred_tr = estimator.predict(X_train)
  y_pred = estimator.predict(X_test)
  print(f'R2 score train: {estimator.score(X_train,y_pred_tr)}')
  print(f'R2 score test: {estimator.score(X_test,y_test)}')
  print(f'mae_train: {mean_absolute_error(y_train, y_pred_tr)}')
  print(f'mse_train: {mean_squared_error(y_train, y_pred_tr)}')
  print(f'mae_test: {mean_absolute_error(y_test, y_pred)}')
  print(f'mse_test: {mean_squared_error(y_test, y_pred)}')
  f, (ax1, ax2) = plt.subplots(1,2)
  ax1.scatter(x=X_test[:,0],y=y_test[:,0])
  ax1.scatter(x=X_test[:,0],y=y_pred[:,0], color='r')
  ax1.set_title('Depth')
  ax2.scatter(x=X_test[:,1],y=y_test[:,1])
  ax2.scatter(x=X_test[:,1],y=y_pred[:,1], color='r')
  ax2.set_title('Width');

knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
lr = LinearRegression()
tree_reg = DecisionTreeRegressor()
mlp = MLPRegressor(hidden_layer_sizes=(500,),max_iter=10000)
rf = RandomForestRegressor(n_estimators=500,n_jobs=-1)

models = [knn,lr,tree_reg,mlp,rf]

for model in models:
  test_model(model)
  print()

"""Видим, что лучше всего работают методы основанные на деревьях решений

Кроссвалидация c поиском по сетке для подбора параметров
"""

mae = make_scorer(mean_absolute_error)

params = {
    'n_estimators': [100,300,500,700],
    'max_depth': [1,3,5,7,10],
    'min_samples_leaf': [1,2,3,5]
}

gs = GridSearchCV(RandomForestRegressor(random_state=1), params, scoring=mae, cv=5, n_jobs=-1)

gs.fit(X_scal, y)

gs.best_estimator_

gs.best_score_

"""Малый обучающий набор данных, поэтому результаты при кроссвалидации отличаются от полученных ранее

Сохраняем модель
"""

import pickle

pickle.dump(gs.best_estimator_, open('model.pkl', 'wb'))