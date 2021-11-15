# -*- coding: utf-8 -*-
"""Additional Classifiers Testing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16E3avMbgZz1YnYvo1uuc4HLzgHZiaDCQ

<center><h1>Mini Project 2 - Bernoulli Naïve Bayes</h1></center>
This file consists two parts:  

In the first part, it measures the accuracies and time spent of the Logistic Regression based on the output of 5 different classifiers. The effect of data normalization is also measured.   

In the second part, it measures the accuracies and time spent of the remaining classifiers given that a TF-IDF vectorizer is used. In the end of the second part, we also did some tests on the Bernoulli Naïve Bayes implemented by sklearn upon all different vectorizers.

<h3>Team Members:</h3>
<center>
Yi Zhu, 260716006<br>
Fei Peng, 260712440<br>
Yukai Zhang, 260710915
</center>
"""

from google.colab import drive
drive.mount('/content/drive')

# make path = './' in-case you are running this locally
path = '/content/drive/My Drive/ECSE_551_F_2020/Mini_Project_02/'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.pipeline import make_pipeline

!pip install nltk
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

"""Additional classifiers:  
1. Logistic Regression
2. Multinomial Naïve Bayes
3. Support Vector Machine
4. Random Forest
5. Decision Tree
6. Ada Boost
7. k-Neighbors
8. Neural Network
"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

reddit_dataset = pd.read_csv(path+"train.csv")
reddit_test = pd.read_csv(path+"test.csv")

X = reddit_dataset['body']
y = reddit_dataset['subreddit']

"""# Define Vectorizer
### (To vectorize the text-based data to numerical features)

1. CountVectorizer  
1) Use "CountVectorizer" to transform text data to feature vectors.  
2) Normalize your feature vectors
"""

def count_vectorizer(X_train, X_test, normalize=True):
    vectorizer = CountVectorizer()
    vectors_train = vectorizer.fit_transform(X_train)
    vectors_test = vectorizer.transform(X_test)

    if normalize:
        normalizer_train = Normalizer().fit(X=vectors_train)
        vectors_train = normalizer_train.transform(vectors_train)
        vectors_test = normalizer_train.transform(vectors_test)

    return vectors_train, vectors_test

"""2. CountVectorizer with stop word  
1) Use "CountVectorizer" with stop word to transform text data to vector.  
2) Normalize your feature vectors
"""

def count_vec_with_sw(X_train, X_test, normalize=True, features_5k=False):
    stop_words = text.ENGLISH_STOP_WORDS
    if features_5k:
        vectorizer = CountVectorizer(stop_words=stop_words, max_features=5000)
    else: 
        vectorizer = CountVectorizer(stop_words=stop_words)
    vectors_train_stop = vectorizer.fit_transform(X_train)
    vectors_test_stop = vectorizer.transform(X_test)

    if normalize:
        normalizer_train = Normalizer().fit(X=vectors_train_stop)
        vectors_train_stop= normalizer_train.transform(vectors_train_stop)
        vectors_test_stop = normalizer_train.transform(vectors_test_stop)

    return vectors_train_stop, vectors_test_stop

"""3. TF-IDF  
1) use "TfidfVectorizer" to weight features based on your train set.  
2) Normalize your feature vectors
"""

def tfidf_vectorizer(X_train, X_test, normalize=True):
    tf_idf_vectorizer = TfidfVectorizer()
    vectors_train_idf = tf_idf_vectorizer.fit_transform(X_train)
    vectors_test_idf = tf_idf_vectorizer.transform(X_test)

    if normalize:
        normalizer_train = Normalizer().fit(X=vectors_train_idf)
        vectors_train_idf= normalizer_train.transform(vectors_train_idf)
        vectors_test_idf = normalizer_train.transform(vectors_test_idf)

    return vectors_train_idf, vectors_test_idf

"""4. CountVectorizer with stem tokenizer  
1) Use "StemTokenizer" to transform text data to vector.  
2) Normalize your feature vectors
"""

class StemTokenizer:
     def __init__(self):
       self.wnl =PorterStemmer()
     def __call__(self, doc):
       return [self.wnl.stem(t) for t in word_tokenize(doc) if t.isalpha()]


def count_vec_stem(X_train, X_test, normalize=True):
    vectorizer = CountVectorizer(tokenizer=StemTokenizer())
    vectors_train_stem = vectorizer.fit_transform(X_train)
    vectors_test_stem = vectorizer.transform(X_test)

    if normalize:
        normalizer_train = Normalizer().fit(X=vectors_train_stem)
        vectors_train_stem= normalizer_train.transform(vectors_train_stem)
        vectors_test_stem = normalizer_train.transform(vectors_test_stem)

    return vectors_train_stem, vectors_test_stem

"""5. CountVectorizer with lemma tokenizer  
1) Use "LemmaTokenizer" to transform text data to vector.  
2) Normalize your feature vectors
"""

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


class LemmaTokenizer:
     def __init__(self):
       self.wnl = WordNetLemmatizer()
     def __call__(self, doc):
       return [self.wnl.lemmatize(t,pos =get_wordnet_pos(t)) for t in word_tokenize(doc) if t.isalpha()]


def count_vec_lemma(X_train, X_test, normalize=True):
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer())
    vectors_train_lemma = vectorizer.fit_transform(X_train)
    vectors_test_lemma = vectorizer.transform(X_test)

    if normalize:
        normalizer_train = Normalizer().fit(X=vectors_train_lemma)
        vectors_train_lemma= normalizer_train.transform(vectors_train_lemma)
        vectors_test_lemma = normalizer_train.transform(vectors_test_lemma)

    return vectors_train_lemma, vectors_test_lemma

"""# Measure Accuracies and Time Spent of different classifiers using K-fold Validation

## 1. Logistic Regression

### 1. CountVectorizer
"""

tic = time()
accuracies = []
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = count_vectorizer(X[train_index], X[test_index])
    clf.fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- Logestic Regression + CountVectorizer + Normalize -\nAccuracy: {}%\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

tic = time()
accuracies = []
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = count_vectorizer(X[train_index], X[test_index], False)
    clf.fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- Logestic Regression + CountVectorizer + Unnormalize -\nAccuracy: {}%\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

"""### 2. CountVectorizer with stop word"""

tic = time()
accuracies = []
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = count_vec_with_sw(X[train_index], X[test_index])
    clf.fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- Logestic Regression + CountVectorizer with stop word + Normalize -\nAccuracy: {}%\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

tic = time()
accuracies = []
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = count_vec_with_sw(X[train_index], X[test_index], False)
    clf.fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- Logestic Regression + CountVectorizer with stop word + Unnormalize -\nAccuracy: {}%\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

"""### 3. TF-IDF"""

tic = time()
accuracies = []
clf = LogisticRegression(C=40.0, max_iter=1000, random_state=0)
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = tfidf_vectorizer(X[train_index], X[test_index])
    clf.fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- Logestic Regression + TF-IDF Vectorizer + Normalize -\nAccuracy: {}%\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

tic = time()
accuracies = []
clf = LogisticRegression(C=40.0, max_iter=1000, random_state=0)
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = tfidf_vectorizer(X[train_index], X[test_index], False)
    clf.fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- Logestic Regression + TF-IDF Vectorizer + Unnormalize -\nAccuracy: {}%\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

"""### 4. CountVectorizer with stem tokenizer"""

tic = time()
accuracies = []
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = count_vec_stem(X[train_index], X[test_index])
    clf.fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- Logestic Regression + CountVectorizer with stem tokenizer + Normalize -\nAccuracy: {}%\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

tic = time()
accuracies = []
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = count_vec_stem(X[train_index], X[test_index], False)
    clf.fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- Logestic Regression + CountVectorizer with stem tokenizer + Unnormalize -\nAccuracy: {}%\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

"""### 5. CountVectorizer with lemma tokenizer"""

tic = time()
accuracies = []
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = count_vec_lemma(X[train_index], X[test_index])
    clf.fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- Logestic Regression + CountVectorizer with lemma tokenizer + Normalize -\nAccuracy: {}%\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

tic = time()
accuracies = []
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = count_vec_lemma(X[train_index], X[test_index], False)
    clf.fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- Logestic Regression + CountVectorizer with lemma tokenizer + Unnormalize -\nAccuracy: {}%\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

"""## 2. Multinomial Naïve Bayes"""

tic = time()
accuracies = []
clf = MultinomialNB()
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = tfidf_vectorizer(X[train_index], X[test_index])
    clf.fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- Multinomial Naïve Bayes + TF-IDF Vectorizer + Normalize -\nAccuracy: {}%\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

"""## 3. Support Vector Machine

Linear
"""

tic = time()
accuracies = []
clf = svm.SVC(kernel='linear', gamma='auto', C=1) 
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = tfidf_vectorizer(X[train_index], X[test_index])
    clf.fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- Linear Support Vector Machine + TF-IDF Vectorizer + Normalize -\nAccuracy: {}%\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

"""## 4. Random Forest"""

tic = time()
accuracies = []
clf = RandomForestClassifier(max_depth=2, random_state=0)
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = tfidf_vectorizer(X[train_index], X[test_index])
    clf.fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- Random Forest + TF-IDF Vectorizer + Normalize -\nAccuracy: {}%\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

"""## 5. Decision Tree"""

tic = time()
accuracies = []
clf = DecisionTreeClassifier(random_state=0)
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = tfidf_vectorizer(X[train_index], X[test_index])
    clf.fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- Decision Tree + TF-IDF Vectorizer + Normalize -\nAccuracy: {}%\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

"""## 6. Ada Boost"""

tic = time()
accuracies = []
clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=0)
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = tfidf_vectorizer(X[train_index], X[test_index])
    clf.fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- Ada Boost + TF-IDF Vectorizer + Normalize -\nAccuracy: {}%\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

"""## 7. k-Neighbors"""

tic = time()
accuracies = []
neigh = KNeighborsClassifier(n_neighbors=3)
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = tfidf_vectorizer(X[train_index], X[test_index])
    neigh.fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], neigh.predict(vectors_test)))

print("\t- k-Neighbors + TF-IDF Vectorizer + Normalize -\nAccuracy: {}%\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

"""## 8. Neural Network"""

tic = time()
accuracies = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = tfidf_vectorizer(X[train_index], X[test_index])
    clf = MLPClassifier(random_state=0, max_iter=300).fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- Neural Network + TF-IDF Vectorizer + Normalize -\nAccuracy: {}%\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

"""## 9. Bernoulli Naïve Bayes (Sklearn Version)
<h2>This part is only used to test and compare the performance of the Bernoulli Naïve Bayes implemented by ourselves.</h2>
"""

from sklearn.naive_bayes import BernoulliNB

"""### 1. CountVectorizer"""

tic = time()
accuracies = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = count_vectorizer(X[train_index], X[test_index])
    clf = BernoulliNB().fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- BernoulliNB + CountVectorizer + Normalize -\nAccuracy: {}\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

"""### 2. CountVectorizer with stop word"""

tic = time()
accuracies = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = count_vec_with_sw(X[train_index], X[test_index])
    clf = BernoulliNB().fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- BernoulliNB + CountVectorizer with stop word + Normalize -\nAccuracy: {}\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

"""### 3. CountVectorizer with stop word, max_features = 5000"""

tic = time()
accuracies = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = count_vec_with_sw(X[train_index], X[test_index], features_5k=True)
    clf = BernoulliNB().fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- BernoulliNB + CountVectorizer with stop word, max_features = 5000 + Normalize -\nAccuracy: {}\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

"""### 4. TF-IDF"""

tic = time()
accuracies = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = tfidf_vectorizer(X[train_index], X[test_index])
    clf = BernoulliNB().fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- BernoulliNB + TF-IDF Vectorizer + Normalize -\nAccuracy: {}\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

"""### 5. CountVectorizer with stem tokenizer"""

tic = time()
accuracies = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = count_vec_stem(X[train_index], X[test_index])
    clf = BernoulliNB().fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- BernoulliNB + CountVectorizer with stem tokenizer + Normalize -\nAccuracy: {}\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))

"""### 6. CountVectorizer with lemma tokenizer"""

tic = time()
accuracies = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    vectors_train, vectors_test = count_vec_lemma(X[train_index], X[test_index])
    clf = BernoulliNB().fit(vectors_train, y[train_index])
    accuracies.append(metrics.accuracy_score(y[test_index], clf.predict(vectors_test)))

print("\t- BernoulliNB + CountVectorizer with lemma tokenizer + Normalize -\nAccuracy: {}\tTime Spent: {}s".format(np.mean(accuracies), time()-tic))