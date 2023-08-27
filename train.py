import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import seaborn as sns


image_foler_path = 'DATASET'


images = []
labels = []

for label in os.listdir(image_foler_path):
    for image_path in os.listdir(os.path.join(image_foler_path, label)):
        image = cv2.imread(os.path.join(image_foler_path, label, image_path))
        images.append(image)
        labels.append(label)

# Prétraitement des images
def preprocess_images(images):
    resized_images = [cv2.resize(image, (224, 224)) for image in images]
    gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in resized_images]
    normalized_images = [image / 255.0 for image in gray_images]
    processed_images = np.expand_dims(normalized_images, axis=-1)
    return processed_images

processed_images = preprocess_images(images)

# Encoder les labels
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
labels = to_categorical(labels)

print(labels.shape)
print(processed_images.shape)


batch_size = 32
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(processed_images, labels, test_size=0.2)

# Créer un modèle de réseau de neurones convolutifs (CNN)
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(7, activation='softmax')
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=batch_size)

# Évaluation du modèle
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Prédire les classes
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)


# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)

labels = ['Ascending Triangle', 'Descending Triangle', 'Double Bottom', 'Double Top', 'Falling Wedge', 'Rising Wedge', 'Symmetrical Triangle']

# Afficher la matrice de confusion
plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt="d" ,cmap="Blues", cbar=False, annot_kws={"size": 20}, linewidths=0.2, linecolor="black", xticklabels=labels, yticklabels=labels)
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('confusion.png',bbox_inches='tight',dpi=400, pad_inches=0.1)

# Sauvegarder le modèle pour une utilisation future
model.save('models/crypto_pattern_model.keras')
