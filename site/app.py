# CLIENT FRONT END APPLICATION 1 #
import process_url

from flask import Flask, render_template, url_for, request, redirect
from datetime import datetime

app = Flask(__name__)



@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/process_url.py', methods=['POST', 'GET'])
def process():
    url = request.form["param"]
    print(url)
    toto = process_url.main(url)
    print(toto)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
