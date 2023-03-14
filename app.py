import json
from model import EnsambleModel
from utils import preprocess_image

import numpy as np
from flask import Flask, render_template, request
from PIL import Image

model = EnsambleModel()
model.load('model_files')

# serve the app with flask
app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/DigitRecognition", methods=["POST"])
def predict_digit():
    img = Image.open(request.files["img"]).convert("L")
    img = preprocess_image(img)

    # predict
    probs = model.predict_prob(img)[0] # first prediction because there is only one
    res_json = {"pred": "Err", "probs": []}
    res_json["pred"] = str(np.argmax(probs))
    res_json["probs"] = [p * 100 for p in probs]
    print(res_json)
    return json.dumps(res_json)

def main():
    app.run(host="0.0.0.0", debug=True)

if __name__ == "__main__":
    main()
    
