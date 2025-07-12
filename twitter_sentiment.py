# twitter_sentiment_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import warnings
from wordcloud import WordCloud
from datetime import datetime
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score
from xgboost import XGBClassifier
from tqdm import tqdm

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

warnings.filterwarnings("ignore")

# Load data
train = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\d\d\train_tweet.csv")
test = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\d\d\test_tweets.csv")
print("Checkpoint 1: Loaded data")

# Add tweet length column
train['len'] = train['tweet'].str.len()
test['len'] = test['tweet'].str.len()

# Text cleaning & preprocessing
train_corpus, test_corpus = [], []
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

for tweet in train['tweet']:
    review = re.sub('[^a-zA-Z]', ' ', tweet).lower().split()
    review = ' '.join([ps.stem(word) for word in review if word not in stop_words])
    train_corpus.append(review)

for tweet in test['tweet']:
    review = re.sub('[^a-zA-Z]', ' ', tweet).lower().split()
    review = ' '.join([ps.stem(word) for word in review if word not in stop_words])
    test_corpus.append(review)

print("Checkpoint 2: Preprocessed text")

# Vectorization
cv = CountVectorizer(max_features=1000)
x = cv.fit_transform(train_corpus).toarray()
y = train['label']
x_test = cv.transform(test_corpus).toarray()
print("x shape:", x.shape)
print("x_test shape:", x_test.shape)
print("Checkpoint 3: Vectorized data")

# Train-test split
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.25, random_state=42)

# Standardization
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_valid = sc.transform(x_valid)
x_test = sc.transform(x_test)

# Models to evaluate
models = [
    RandomForestClassifier(n_jobs=-1),
    LogisticRegression(),
    DecisionTreeClassifier(),
    XGBClassifier(n_jobs=-1, verbosity=0)
]

best_model = None
best_score = 0

for model in models:
    try:
        print(f"\nTraining: {model.__class__.__name__}")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_valid)
        print("✅ Model trained and predicted")

        print("Training Accuracy:", model.score(x_train, y_train))
        print("Validation Accuracy:", model.score(x_valid, y_valid))

        try:
            score = f1_score(y_valid, y_pred)
            print("F1 Score:", score)
        except Exception as e:
            print(f"⚠️ F1 Score calculation failed: {e}")
            score = 0

        print("Confusion Matrix:\n", confusion_matrix(y_valid, y_pred))

        if score > best_score:
            best_score = score
            best_model = model

    except Exception as e:
        print(f"❌ Failed on {model.__class__.__name__}: {e}")
print("Checkpoint 4: Trained models")

# Export predictions
try:
    predictions = best_model.predict(x_test)
    ids = test['id'] if 'id' in test.columns else pd.Series(range(len(predictions)), name='id')
    tweets = test['tweet'] if 'tweet' in test.columns else [''] * len(predictions)

    output_df = pd.DataFrame({
        'id': ids,
        'tweet': tweets,
        'label': predictions
    })

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"submission_{timestamp}.csv"
    output_df.to_csv(filename, index=False)

    print(f"\n✅ Test predictions saved to: {filename}")
    print(output_df.head())

    try:
        os.startfile(filename)
    except Exception as e:
        print(f"Note: Could not auto-open file. Reason: {e}")
except Exception as e:
    print(f"❌ Error during export: {e}")

print("✅ ALL DONE — MODEL TRAINED, PREDICTED, AND EXPORTED.")