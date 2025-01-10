from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO

app = Flask(__name__)

model = load_model('dog_cat_classifier.keras')


@app.route('/predict', methods=['POST'])
def predict():
    if 'photo' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['photo']
    try:
        # Use BytesIO to handle the FileStorage object
        img = image.load_img(BytesIO(file.read()), target_size=(150, 150))

        # Preprocess the image
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalize the image

        # Perform prediction
        classes = model.predict(x)
        if classes[0] > 0.5:
            predicted_class = "Dog"
        else:
            predicted_class = "Cat"

        # Return the prediction
        return jsonify({
            'success': True,
            'data': {
                'prediction' : predicted_class
            }
        })
    except Exception as e:
        return jsonify({
            'success' : False,
            'message': str(e)
        }), 500


# Define a route for the upload form
@app.route('/', methods=['GET'])
def upload_form():
    return render_template('index.html')
