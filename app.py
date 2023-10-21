from flask import Flask, request, redirect, url_for, render_template
import keras.models
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = keras.models.load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Load the calorie information
calorie_data = {
    "cookies": 150,
    "noodles": 200,
    "oatmeal": 150,
    "rice": 130,
    "toast": 100,
    "cabbage": 25,
    "strawberry": 50,
    "banana": 105,
    "cauliflower": 25,
    "cucumber": 15,
    "shrimp": 100,
    "fish": 100,
    "chicken breast": 150,
    "egg": 78,
    "sweet sour pork": 300,
}

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)

        # Save file locally
        file.save('uploaded_file.jpg')

        # Load and prepare the image
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open('uploaded_file.jpg').convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Delete the temporary file
        import os
        os.remove('uploaded_file.jpg')

        print("Class Name from Model:", class_name)
        
        # Get the food class name (excluding index)
        food_class = class_name[2:].strip()

        print("Food Class:", food_class)
        print("Is in calorie_data:", food_class.lower() in calorie_data)

        # Look up the calorie count
        if food_class.lower() in calorie_data:
            calorie_count = calorie_data[food_class.lower()]
        else:
            calorie_count = "Calorie information not available."

        # Return prediction, confidence score, and calorie information
        return f"Class: {food_class}, Confidence Score: {confidence_score}, Calories: {calorie_count}"

    # Display an HTML form for uploading a file
    return render_template('index.html', title='Upload File')

if __name__ == '__main__':
    app.run(debug=True)
