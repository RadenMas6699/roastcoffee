import tensorflow as tf
import numpy as np
import os
import cv2
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Load the TFLite model
model_path = 'model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the test image
def preprocess_image(img_path, img_size):
    img_array = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    new_array = cv2.resize(img_array, (img_size, img_size))

    # Convert the image to float32 and normalize
    Xt = np.array(new_array, dtype=np.float32).reshape(-1, img_size, img_size, 3) / 255.0
    return Xt

# Process the prediction results
def process_prediction(prediction):
    CATEGORIES = ['Dark', 'Green', 'Light', 'Medium']
    predicted_class_index = np.argmax(prediction)
    predicted_class = CATEGORIES[predicted_class_index]
    class_percentages = (prediction[0] * 100).tolist()  # Convert to Python list

    # Round the percentages to two decimal places
    rounded_percentages = [round(percentage, 2) for percentage in class_percentages]

    # Create a dictionary with category names and rounded percentages
    result = {CATEGORIES[i]: percentage for i, percentage in enumerate(rounded_percentages)}
    return predicted_class, result

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "Work"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)

        # Preprocess the image
        IMG_SIZE = 70
        Xt = preprocess_image(file_path, IMG_SIZE)

        # Perform inference with the TFLite model
        interpreter.set_tensor(input_details[0]['index'], Xt)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # Process the prediction results
        predicted_class, class_percentages = process_prediction(prediction)

        # Remove the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({'predicted_class': predicted_class, 'class_percentages': class_percentages})

if __name__ == '__main__':
    app.run(debug=False)