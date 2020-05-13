import io
import os
import base64
import pickle
import numpy as np
from PIL import Image
from flask import Flask, request, redirect, render_template, jsonify
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.inception_v3 import preprocess_input
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import load_model

with open("./model/wordtoix.pkl", "rb") as fp:
  wordtoix = pickle.load(fp)

with open("./model/ixtoword.pkl", "rb") as fp:
  ixtoword = pickle.load(fp)

enocoding_model = load_model('./model/encoding_model.h5', compile = False)

prediction_model = load_model('./model/final_model.h5')

def preprocess(PIL_image):
    resized_image = PIL_image.resize((299, 299))
    x = image.img_to_array(resized_image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode(image):
    image = preprocess(image)
    fea_vec = enocoding_model.predict(image)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec

def greedySearch(photo):
    in_text = 'startseq'
    for i in range(34):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=34)
        yhat = prediction_model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods = ['GET','POST'])
def base():
    if request.method == 'GET':
        return render_template("base.html")
    else:
        if 'input_image' not in request.files:
            print("No file part")
            return redirect(request.url)
        file = request.files['input_image']
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            img = Image.open(request.files['input_image'].stream)
            output_str = greedySearch(encode(img).reshape((1, 2048)))
            buff = io.BytesIO()
            img.save(buff, format="JPEG")
            new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
            print(output_str)
            
    if request.form.get('platform') == 'website':
        return render_template('base.html', caption = output_str, img = new_image_string)
    else:
        return jsonify({'caption': output_str})

if __name__ == "__main__":
    app.secret_key = 'qwertyuiop1234567890'
    port = int(os.environ.get('PORT', 33507))
    app.run(debug=True,host='0.0.0.0',port=port)
    print("APP IS RUNNING")
