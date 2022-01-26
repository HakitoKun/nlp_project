# CLIENT FRONT END APPLICATION 1 #
import process_url

from flask import Flask, render_template, url_for, request, redirect, jsonify
from datetime import datetime

from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
model = TFAutoModelForSeq2SeqLM.from_pretrained("../checkpoint-110000/", from_pt=True)
tokenizer = AutoTokenizer.from_pretrained("t5-base")


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/process_url.py', methods=['POST', 'GET'])
def process():
    url = request.form["param"]
    print(url)
    toto = process_url.main(url, model, tokenizer)
    print(toto)
    #return render_template('index.html', resume=toto)
    return jsonify(result=toto)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
