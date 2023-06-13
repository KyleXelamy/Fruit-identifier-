import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load and preprocess the image
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_preprocessed = preprocess_input(img_array)
    img_expanded = np.expand_dims(img_preprocessed, axis=0)
    return img_expanded

# Make predictions on the image
def predict_image(image_path):
    img = load_and_preprocess_image(image_path)
    predictions = model.predict(img)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

# Classify user-provided image
def classify_user_image(image_path):
    if not os.path.isfile(image_path):
        print("The provided image file does not exist.")
        return

    predictions = predict_image(image_path)
    if len(predictions) == 0:
        print("The fruit in the image could not be identified.")
    else:
        print("Predictions for the provided image:")
        for prediction in predictions:
            print(f"{prediction[1]}: {prediction[2]}")

# Path to the folder containing the images
folder_path = "fruits"

# Iterate over the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        predictions = predict_image(image_path)
        
        print(f"Predictions for {filename}:")
        for prediction in predictions:
            print(f"{prediction[1]}: {prediction[2]}")
        print()

# Prompt the user to input an image
user_image_path = input("Enter the path to an image: ")
classify_user_image(user_image_path)
