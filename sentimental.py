import pandas as pd 
import string 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

# 1 Load the dataset
data =pd.read_csv("IMDB Dataset.csv")

print(data.head())
print(data.shape)

# 2 Convert the labels to numbers 

data['sentiment'] =data['sentiment'].map({'positive' : 1,'negative':0})

# 3 Text Preprocessing Function 
def preprocess(text):

  # Lower case
  text =text.lower()

  # Tokenization

  tokens = text.split()

  tokens=[word.strip(string.punctuation) for word in tokens]

  text =" ".join(tokens)

  return text

# Apply Preprocessing

data['clean_review'] = data['review'].apply(preprocess)

# Convert text to numbers -> TF-IDF Vectors

vectorizer = TfidfVectorizer(max_features=5000)

X= vectorizer.fit_transform(data['clean_review'])

y =data['sentiment']

# Train test split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train models 
lr_model=LogisticRegression()
nb_model=MultinomialNB()

lr_model.fit(X_train,y_train)
nb_model.fit(X_train,y_train)

#Predictions 
lr_pred= lr_model.predict(X_test)
nb_pred=nb_model.predict(X_test)

# Evaluation
print("Logistic Regression Accuracy :")
print(accuracy_score(y_test,lr_pred))

print("Confusion Matrix")
print(accuracy_score(y_test,nb_pred))

print("\n Confusion matrix (Logistic Regression):")
print(confusion_matrix(y_test,lr_pred))

print("\n Classification Report")
print(classification_report(y_test,lr_pred))

import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_test,lr_pred)

sns.heatmap(cm,annot=True,fmt='d')

plt.title("Cnofusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
