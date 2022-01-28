# CLIENT FRONT END APPLICATION 1 #

from flask import Flask, render_template, request, jsonify
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

import process_url

app = Flask(__name__)
model = TFAutoModelForSeq2SeqLM.from_pretrained("../checkpoint-110000/", from_pt=True)
tokenizer = AutoTokenizer.from_pretrained("t5-base")


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')


@app.route('/process_url.py', methods=['POST', 'GET'])
def process():
    url = request.form["param"]
    length = int(request.form["length"])
    print(url, length)
    generated_text = process_url.main(url, model, tokenizer, length)
    print(generated_text)
    return jsonify(result=generated_text)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5100, debug=True)
