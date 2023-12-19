import os
from PIL import Image
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

# Set the environment variable before importing TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load your CIFAR-10 model
model = tf.keras.models.load_model('model_cifer10.h5')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Preprocess the image
        class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        img = Image.open(file.stream).resize((32, 32))
        img_array = np.array(img)
        img_array = img_array.reshape((1, 32, 32, 3))

        # Make prediction
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = prediction[0, class_index]

        # Map class index to class name
        class_name = class_names[class_index]

        return jsonify({'class': class_name, 'confidence': float(confidence)})

if __name__ == '__main__':
    app.run(debug=True)
