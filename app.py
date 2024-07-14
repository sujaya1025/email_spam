from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model and CountVectorizer
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))

# Define the feature extraction process
def extract_features(email):
    return cv.transform([email])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form['email']
    features = extract_features(email)
    prediction = model.predict(features)[0]
    result = 'spam' if prediction == 1 else 'ham'
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
