import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np


def load_model(model_path):
    return keras.models.load_model(model_path)

def predict(model, image):
    return model.predict(image)

def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    normalized_image = gray_image / 255.0
    processed_image = np.expand_dims(normalized_image, axis=-1)
    return processed_image

def predict_image(model, image_path, class_labels):
    image = cv2.imread(image_path)
    processed_image = preprocess_image(image)
    print(processed_image.shape)
    prediction = predict(model, np.expand_dims(processed_image, axis=0)) 
    predicted_class_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_class_index]

    print("Prédiction : ", predicted_label)
    print("Confiance : ", prediction[0][predicted_class_index])

    return predicted_label

model = load_model('models/crypto_pattern_model.keras')

class_labels = ['Ascending Triangle', 'Descending Triangle', 'Double Bottom', 'Double Top', 'Falling Wedge', 'Rising Wedge', 'Symmetrical Triangle']


image_path = 'image.png'


predicted_label = predict_image(model, image_path, class_labels)
