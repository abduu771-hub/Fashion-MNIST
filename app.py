from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
model = load_model('model/fashion_model.h5')
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    """Convert uploaded image to Fashion MNIST format"""
    img = Image.open(image_path).convert('L')  # Grayscale
    img = img.resize((28, 28))                # Resize
    img_array = np.array(img) / 255.0         # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)  # Add batch/channel dims
    return img_array
from flask import jsonify

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        img_array = preprocess_image(save_path)
        preds = model.predict(img_array)
        predicted_class = class_names[np.argmax(preds)]
        confidence = float(np.max(preds)) * 100

        return jsonify({
            'prediction': {
                'class': predicted_class,
                'confidence': f"{confidence:.2f}%",
                'all_predictions': dict(zip(class_names, [f"{p*100:.1f}%" for p in preds[0]]))
            }
        })

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    uploaded_image = None
    
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
            
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="Empty filename")
            
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            
            # Predict
            img_array = preprocess_image(save_path)
            preds = model.predict(img_array)
            predicted_class = class_names[np.argmax(preds)]
            confidence = float(np.max(preds)) * 100
            
            # Pass results to template
            uploaded_image = url_for('static', filename=f'uploads/{filename}')
            prediction = {
                'class': predicted_class,
                'confidence': f"{confidence:.2f}%",
                'all_predictions': dict(zip(class_names, [f"{p*100:.1f}%" for p in preds[0]]))
            }

    return render_template('index.html', prediction=prediction, uploaded_image=uploaded_image)

if __name__ == '__main__':
    app.run(debug=True)