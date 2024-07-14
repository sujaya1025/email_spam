import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load the dataset
ds = pd.read_csv("C:/Users/P.SUJAYA RAGHAVI/Downloads/datasets/archive (5).zip", encoding='latin1')

# Preprocess the dataset
ds.replace({'spam': 1, 'ham': 0}, inplace=True)

# Extract features and labels
X = ds['v2']
y = ds['v1']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert the text data into numerical features using CountVectorizer
cv = CountVectorizer(max_features=5000)
X_trainCV = cv.fit_transform(X_train.values)

# Train the model
model = MultinomialNB()
model.fit(X_trainCV, y_train)

# Save the trained model and CountVectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(cv, open('cv.pkl', 'wb'))

# Evaluate the model
X_testCV = cv.transform(X_test)
y_pred = model.predict(X_testCV)
accuracy = (y_pred == y_test).mean()
print(f'Model accuracy: {accuracy * 100:.2f}%')
