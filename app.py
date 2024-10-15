# app.py
from flask import Flask, request, render_template
import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load the models
cnn_model = load_model('cnn_model.h5')
lstm_model = load_model('lstm_model.h5')

# Load the recipe data
data = pd.read_csv('merged_recipes_with_images.csv')
recipe_images = {}
for index, row in data.iterrows():
    recipe = row['Recipe']
    images = row['Image Paths'].split(', ')
    recipe_images[recipe] = images

y_recipes_unique = list(recipe_images.keys())

# Define image processing function
def process_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Define prediction function
def predict_recipe(image_path):
    img_array = process_image(image_path)
    cnn_prediction = cnn_model.predict(img_array)
    predicted_class = np.argmax(cnn_prediction, axis=1)
    predicted_recipe = y_recipes_unique[predicted_class[0]]
    return predicted_recipe

# Define upload directory
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message="No file uploaded.")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message="No file selected.")
    
    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    # Make prediction
    predicted_recipe = predict_recipe(file_path)
    
    return render_template('result.html', recipe=predicted_recipe)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
