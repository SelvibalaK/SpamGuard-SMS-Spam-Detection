# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:19:59 2024

@author: Selvibala
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import random
data = pd.read_csv(r'C:\Users\Selvibala\Downloads\spam.csv', encoding='latin-1')
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(data["v2"])
y = data["v1"]
model = LogisticRegression()
model.fit(X_tfidf, y)
random_index = random.randint(0, len(data) - 1)
selected_text = data.loc[random_index, "v2"]
print(f"Selected Text:\n{selected_text}\n")
X_selected_tfidf = tfidf_vectorizer.transform([selected_text])
predicted_label = model.predict(X_selected_tfidf)[0]
if predicted_label == "ham":
    print("Predicted Label: Ham")
else:
    print("Predicted Label: Spam")
accuracy = model.score(X_tfidf, y)
print(f"Model Accuracy: {accuracy:.2f}")
