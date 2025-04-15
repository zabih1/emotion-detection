import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import yaml


train_data = pd.read_csv("data/features/train_bow.csv")

X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:, -1].values


n_estimators = yaml.safe_load(open("params.yaml", 'r'))['model_building']['n_estimators']
learning_rate = yaml.safe_load(open("params.yaml", 'r'))['model_building']['learning_rate']

clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
clf.fit(X_train, y_train)


pickle.dump(clf, open('model.pkl', 'wb'))
