from crypt import methods
from pyexpat import features, model
import numpy as np;
from flask import Flask, request , render_template
import joblib


app = Flask(__name__)
model=joblib.load("studentmarkpredictor")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict' , methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value  = np.array(input_features)

    output = model.predict([features_value])[0][0].round(2)

    return render_template('index.html', prediction_text = 'you will get [{} %] marks ,when you study for {output} hours')


if __name__ =="__main__":
    app.run()
