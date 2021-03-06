# -*- coding: utf-8 -*-
"""Data Preprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bRmLzkFmGP7xgUOUI6iv_p6dW1hX6R6P

<center><h1>Mini Project 2 - Bernoulli Naïve Bayes</h1>
<h3>Data Preprocessing</h3>
<h4>This file performs some of the operations on Data Preprocessing and Analysis.</h4></center>

<h3>Team Members:</h3>
<center>
Yi Zhu, 260716006<br>
Fei Peng, 260712440<br>
Yukai Zhang, 260710915
</center>

# Importations
"""

from google.colab import drive
drive.mount('/content/drive')

# make path = './' in-case you are running this locally
path = '/content/drive/My Drive/ECSE_551_F_2020/Mini_Project_02/'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy import stats
from google.colab import files
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

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

"""# Data Preprocessing"""

reddit_dataset = pd.read_csv(path+"train.csv")
reddit_test = pd.read_csv(path+"test.csv")

X = reddit_dataset['body']
y = reddit_dataset['subreddit']

class Data_Processing:
    def __init__(self, data, name='New Data'):
        self.data = data
        self.name = name

    def show_y_dist(self, ydata):
        plt.figure(figsize=(8,4))
        plt.subplot(111), sns.countplot(x='subreddit', data=ydata)
        plt.title('Distribution of Subreddit in {}'.format(self.name))
        plt.savefig("Distribution of Subreddit in {}.png".format(self.name), dpi = 1200)
        files.download("Distribution of Subreddit in {}.png".format(self.name))
        plt.show()

data_analysis = Data_Processing(reddit_dataset.values, 'train.csv')
data_analysis.show_y_dist(reddit_dataset)

# calculate the data entropy
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()  # encoder for classes
le.fit(y)
y_label = le.transform(y)
n_k = len(le.classes_)
N = len(y)
theta_k = np.zeros(n_k)  # probability of class k
# compute theta values
for k in range(n_k):
    count_k = (y_label==k).sum()
    theta_k[k] = count_k / N

from scipy.stats import entropy
print("Data entropy is", entropy(theta_k, base=2))

"""# Define Vectorizer 
(To vectorize the text-based data to numerical features)

1. CountVectorizer  
1) Use "CountVectorizer" to transform text data to feature vectors.  
2) Normalize your feature vectors
"""

def count_vectorizer(X_train):
    vectorizer = CountVectorizer()
    vectors_train = vectorizer.fit_transform(X_train)
    return vectors_train

"""2. CountVectorizer with stop word  
1) Use "CountVectorizer" with stop word to transform text data to vector.  
2) Normalize your feature vectors
"""

def count_vec_with_sw(X_train):
    stop_words = text.ENGLISH_STOP_WORDS
    vectorizer = CountVectorizer(stop_words=stop_words)
    vectors_train_stop = vectorizer.fit_transform(X_train)
    return vectors_train_stop

"""3. TF-IDF  
1) use "TfidfVectorizer" to weight features based on your train set.  
2) Normalize your feature vectors
"""

def tfidf_vectorizer(X_train):
    tf_idf_vectorizer = TfidfVectorizer()
    vectors_train_idf = tf_idf_vectorizer.fit_transform(X_train)
    return vectors_train_idf

"""4. CountVectorizer with stem tokenizer  
1) Use "StemTokenizer" to transform text data to vector.  
2) Normalize your feature vectors
"""

class StemTokenizer:
     def __init__(self):
       self.wnl =PorterStemmer()
     def __call__(self, doc):
       return [self.wnl.stem(t) for t in word_tokenize(doc) if t.isalpha()]


def count_vec_stem(X_train):
    vectorizer = CountVectorizer(tokenizer=StemTokenizer())
    vectors_train_stem = vectorizer.fit_transform(X_train)
    return vectors_train_stem

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


def count_vec_lemma(X_train):
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer())
    vectors_train_lemma = vectorizer.fit_transform(X_train)
    return vectors_train_lemma

"""# Measure the time required for each vectorizer to perform vectorization

### 1. CountVectorizer
"""

tic = time()
X_vec = count_vectorizer(X)
print("\t\t- Count Vectorizer - \nfeature number: ", X_vec.shape[1], "\t\tTime spent: ", time()-tic, "s.")

"""### 2. CountVectorizer with stop word"""

tic = time()
X_vec = count_vec_with_sw(X)
print("\t\t- Count Vectorizer with stop word - \nfeature number: ", X_vec.shape[1], "\t\tTime spent: ", time()-tic, "s.")

"""### 3. TF-IDF"""

tic = time()
X_vec = tfidf_vectorizer(X)
print("\t\t- TF-IDF Vectorizer - \nfeature number: ", X_vec.shape[1], "\t\tTime spent: ", time()-tic, "s.")

"""### 4. CountVectorizer with stem tokenizer"""

tic = time()
X_vec = count_vec_stem(X)
print("\t\t- CountVectorizer with stem tokenizer - \nfeature number: ", X_vec.shape[1], "\t\tTime spent: ", time()-tic, "s.")

"""### 5. CountVectorizer with lemma tokenizer"""

tic = time()
X_vec = count_vec_lemma(X)
print("\t\t- CountVectorizer with lemma tokenizer - \nfeature number: ", X_vec.shape[1], "\t\tTime spent: ", time()-tic, "s.")