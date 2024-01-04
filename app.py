# importing packages
from flask import Flask, render_template, request, jsonify
import pickle
import train
import numpy as np
app = Flask(__name__,template_folder='templates', static_folder='./Frontend')

# importing process_text
from train import process_text
model = pickle.load(open('model.pkl','rb'))


# defining route to index.html
@app.route('/')
def welcome():
    return render_template('index.html')


@app.route('/index.html')
def viewindex():
    return render_template('index.html')


# defining route to single-product.html
@app.route('/single-product.html')
def singleproductpage():
    return render_template('single-product.html')

# defining route to get form data after submitting
@app.route('/predict', methods=['POST','GET'])
def predict():
    # return jsonify(request.form)
    feature = request.form.values()
    # feature = request.form
    print(feature)
    prediction = model.predict_proba(feature)
    output = {'prediction': '{0:.{1}f}'.format(prediction[0][1], 2)}
    # output = '{0:.{1}f}'.format(prediction[0][1], 2)

    return jsonify(output)
    pred = ""
    if output>str(0.5):
        return jsonify('fake')
        # return render_template('single-product.html', pred='The probability of this review is fake is {}'.format(output))
    else:
        return jsonify('genuine')
        # return render_template('single-product.html', pred='The probability of this review is genuine is {}'.format(output))


if __name__ == '__main__':
    app.debug = True
    app.run()