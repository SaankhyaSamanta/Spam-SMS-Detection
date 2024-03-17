import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,accuracy_score

#Multinomial SVM
def MultiSVC(x_train, y_train, x_test, y_test):
    classifier = SVC(kernel='poly', degree=3, C=1.0)
    return Classification(classifier, x_train, y_train, x_test, y_test)

#Multinomial Naive Bayes
def MultinomialNaiveBayes(x_train, y_train, x_test, y_test):
    classifier = MultinomialNB()
    return Classification(classifier, x_train, y_train, x_test, y_test)

#Logistic Regression
def LogReg(x_train, y_train, x_test, y_test):
    classifier = LogisticRegression(max_iter=1000)
    return Classification(classifier, x_train, y_train, x_test, y_test)
    
def Classification(classifier, x_train, y_train, x_test, y_test):
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    report = classification_report(y_test, predictions)
    return predictions, report

# GFG

df = pd.read_csv("spam.csv",encoding='latin-1')
print(df.head())

df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df = df.rename(columns={'v1':'label','v2':'Text'})
df['label_enc'] = df['label'].map({'ham':0,'spam':1})
print(df.head())
sns.countplot(x=df['label'])
plt.show()

X, y = np.asanyarray(df['Text']), np.asanyarray(df['label_enc'])
new_df = pd.DataFrame({'Text': X, 'label': y})
X_train, X_test, y_train, y_test = train_test_split(
    new_df['Text'], new_df['label'], test_size=0.2, random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape

tfidf_vec = TfidfVectorizer().fit(X_train)
X_train_vec,X_test_vec = tfidf_vec.transform(X_train),tfidf_vec.transform(X_test)

content = pd.DataFrame()
file = open("Spam SMS Prediction Reports.txt", 'w')

print("Naive Bayes: \n")
Pred, Rep = MultinomialNaiveBayes(X_train_vec, y_train, X_test_vec, y_test)
content["Naive Bayes Predictions"] = Pred
print(Rep)
file.write("Naive Bayes \n")
file.write(Rep)
file.write('\n')

print("Support Vector Machine: \n")
Pred, Rep = MultiSVC(X_train_vec, y_train, X_test_vec, y_test)
content["SVM Predictions"] = Pred
print(Rep)
file.write("Support Vector Machine \n")
file.write(Rep)
file.write('\n')

print("Logistic Regression: \n")
Pred, Rep = LogReg(X_train_vec, y_train, X_test_vec, y_test)
content["Logistic Regression Predictions"] = Pred
print(Rep)
file.write("Logistic Regression \n")
file.write(Rep)
file.write('\n')

file.close()

content.to_csv('Spam SMS Prediction.csv', index=False)

