# phishing_detector.py

# ==============================
# PHISHING EMAIL DETECTION MODEL
# ==============================

import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================
# 1. LOAD DATASET
# ==============================

# Download dataset from:
# https://www.kaggle.com/datasets/subhajournal/phishingemails

df = pd.read_csv("phishing_email.csv")

# Expected columns: 'text' and 'label'
# label: phishing = 1, safe = 0

print(df.head())


# ==============================
# 2. CUSTOM FEATURE EXTRACTOR
# ==============================

class EmailFeatureExtractor(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        
        for text in X:
            text = str(text)
            
            # URL count
            urls = len(re.findall(r"http[s]?://", text))
            
            # Suspicious keywords
            keywords = len(re.findall(r"(urgent|verify|password|login|bank|account)", text.lower()))
            
            # Number of special characters
            special_chars = len(re.findall(r"[!@#$%^&*()]", text))
            
            # Length of email
            length = len(text)
            
            features.append([urls, keywords, special_chars, length])
        
        return np.array(features)


# ==============================
# 3. FEATURE PIPELINE
# ==============================

tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

custom_features = EmailFeatureExtractor()

combined_features = FeatureUnion([
    ("tfidf", tfidf),
    ("custom", custom_features)
])


# ==============================
# 4. SPLIT DATA
# ==============================

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# 5. TRANSFORM FEATURES
# ==============================

X_train_features = combined_features.fit_transform(X_train)
X_test_features = combined_features.transform(X_test)


# ==============================
# 6. TRAIN MODEL
# ==============================

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_features, y_train)


# ==============================
# 7. PREDICTION
# ==============================

y_pred = model.predict(X_test_features)


# ==============================
# 8. EVALUATION
# ==============================

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ==============================
# 9. TEST WITH CUSTOM EMAIL
# ==============================

def predict_email(email_text):
    features = combined_features.transform([email_text])
    prediction = model.predict(features)[0]
    
    if prediction == 1:
        return "⚠️ Phishing Email"
    else:
        return "✅ Safe Email"


# Example
sample_email = "URGENT: Your bank account is compromised! Click http://fake-link.com to verify"

print("\nSample Prediction:", predict_email(sample_email))