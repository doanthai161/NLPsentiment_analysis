import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
np.random.seed(0)

train = pd.read_csv('dataset/sentiment_analysis/Train.csv')
test = pd.read_csv('dataset/sentiment_analysis/Test.csv')

#print(train.head())
#print(train.info())

# print(train['label'].value_counts())
# train['label'].value_counts().plot.bar()
# plt.show()

import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
import string
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords

def cleaning(text):
    text = text.lower()
    text =  re.sub(r'@\S+', '',text)  # remove twitter handles
    text =  re.sub(r'http\S+', '',text) # remove urls
    text =  re.sub(r"[^a-zA-Z']", ' ',text) # only keeps characters
    text =  re.sub(r'\s+[a-zA-Z]\s+', ' ', text+' ')  # keep words with length>1 only
    text = "".join([i for i in text if i not in string.punctuation])
    words = word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')   # remove stopwords
    text = " ".join([i for i in words if i not in stopwords])
    # text= re.sub("\s[\s]+", " ",text).strip()
    text= re.sub("\s[\s]+", " ",text).strip() # remove repeated/leading/trailing spaces

    return text


from nltk.stem.wordnet import WordNetLemmatizer

#lemmatization
def lemm(data):
    wordnet = WordNetLemmatizer()
    lemmanized = []
    for i in range(len(data)):
        lemmed = []
        words = word_tokenize(data['text'].iloc[i])
        for j in words:
            lemmed.append(wordnet.lemmatize(j))
        lemmanized.append(lemmed)

    data['lemmanized'] = lemmanized
    data['text'] = data['lemmanized'].apply(' '.join)
    data = data.drop('lemmanized', axis = 1)
    return data

train['text'] = train['text'].apply(cleaning)
test['text'] = test['text'].apply(cleaning)
#print(train.head())

train = lemm(train)
test = lemm(test)
#print(test.head())

from sklearn.feature_extraction.text import TfidfVectorizer

tfid = TfidfVectorizer()
X_train = tfid.fit_transform(train["text"])
X_test = tfid.transform(test["text"])
y_train = train["label"]
y_test = test["label"]
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=50)

# use fit_transform on our standardized training features
X_train_pca = svd.fit_transform(X_train)
# use transform on our standardized test features
X_test_pca = svd.transform(X_test)

# look at the new shape of the transformed matrices
print('Training features matrix is: ', X_train_pca.shape)
print('Test features matrix is: ', X_test_pca.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

lr = LogisticRegression()
lr.fit(X_train_pca,y_train)
y_lr_pred = lr.predict(X_test_pca)
print('logitics ',accuracy_score(y_lr_pred,y_test))

from sklearn.naive_bayes import GaussianNB
gs = GaussianNB()
gs.fit(X_train_pca, y_train)
y_gs_pred = gs.predict(X_test_pca)
print('gaussion ',accuracy_score(y_test, y_gs_pred))

from sklearn.svm import SVC
SVM = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_train_pca, y_train)
y_svm_pred = SVM.predict(X_test_pca)
print('svm ',accuracy_score(y_svm_pred,y_test))

import lightgbm as ltb
clf = ltb.LGBMClassifier(force_col_wise=True)
clf.fit(X_train_pca, y_train)
preds = clf.predict(X_test_pca)
print(accuracy_score(preds,y_test))

from sklearn.ensemble import RandomForestClassifier

rdfr = RandomForestClassifier(min_samples_split=2,bootstrap=False, max_depth=None, random_state=42,n_jobs=-1, max_features='sqrt')
rdfr.fit(X_train_pca, y_train)
y_rdfr_pred = rdfr.predict(X_test_pca)
print('forest',accuracy_score(y_test, y_rdfr_pred))
print('forest',confusion_matrix(y_test, y_rdfr_pred))