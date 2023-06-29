from training import training
from prediction import prediction

from flask import Flask, request, render_template, redirect, request

app = Flask(__name__)


@app.route("/")
def Home():
    return render_template('index.html')

@app.route("/prediction", methods=['POST', 'GET'])
def Prediction():
    result = None
    if request.method == 'POST':
        messages = request.form['messages']
        result = prediction(new_data=messages)
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
