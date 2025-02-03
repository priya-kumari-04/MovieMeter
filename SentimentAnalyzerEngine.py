from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model_path = 'sentimentanalysis\\models\\model\\classifier.pkl'
classifier = joblib.load(model_path)

def predictfunc(review):    
    prediction = classifier.predict(review)
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    return prediction[0], sentiment

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        content = request.form.get('review')  # Safely get the review input
        if content:
            review = pd.Series(content)
            prediction, sentiment = predictfunc(review)
            return render_template("predict.html", pred=prediction, sent=sentiment)
        else:
            return render_template("home.html", error="Please provide a review.")

if __name__ == '__main__':
    app.run(host='0.0.0.0')
