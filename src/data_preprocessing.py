import numpy as np
import pandas as pd
import re
import nltk
import string 
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import os

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, roc_auc_score


nltk.download('wordnet')
nltk.download('stopwords')


# fetch data from the raw folder

train_data = pd.read_csv("data/raw/test.csv")
test_data = pd.read_csv("data/raw/test.csv")


def lemmatization(text):
    lemmatizater = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizater.lemmatize(y) for y in text]
    return " ".join(text)
    

def remove_stop_words(text):
    text = "".join([i for i in text if not i.isdigit()])
    return text

def removing_numbers(text):
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

def lower_case(text):
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace(":", "")

    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

    

def removing_punctuations(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    df.content=df.content.apply(lambda content : lower_case(content))
    df.content=df.content.apply(lambda content : remove_stop_words(content))
    df.content=df.content.apply(lambda content : removing_numbers(content))
    df.content=df.content.apply(lambda content : removing_punctuations(content))
    df.content=df.content.apply(lambda content : removing_urls(content))
    df.content=df.content.apply(lambda content : lemmatization(content))
    return df

train_processed_data = normalize_text(train_data)
test_processed_data = normalize_text(test_data)


data_path = os.path.join("data", 'processed')
os.makedirs(data_path, exist_ok=True)

train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
train_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)




