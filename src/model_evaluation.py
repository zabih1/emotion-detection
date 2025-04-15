from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import pandas as pd
import numpy as np
import pickle
import json

test_data = pd.read_csv("data/features/test_bow.csv")
X_test = test_data.iloc[:,0:-1].values
y_test = test_data.iloc[:, -1].values


clf = pickle.load(open("model.pkl",'rb'))



y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)


metrics_dict = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "AUC": auc
}


with open("metrics.json", 'w') as f:

    json.dump(metrics_dict, f, indent=4)