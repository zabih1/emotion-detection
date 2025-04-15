import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pickle


train_data = pd.read_csv("data/features/train_bow.csv")

X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:, -1].values


clf = GradientBoostingClassifier(n_estimators=50)
clf.fit(X_train, y_train)


pickle.dump(clf, open('model.pkl', 'wb'))
