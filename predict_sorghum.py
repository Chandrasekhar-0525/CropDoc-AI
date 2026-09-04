import tensorflow as tf
import numpy as np
import json

# Load model
model = tf.keras.models.load_model("sorghum_disease_model.keras")

# Load class names
with open("sorghum_class_names.json", "r") as f:
    class_names = json.load(f)

# Ask for image
image_path = input("Enter image path: ")

# Load image
image = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

# Convert image to array
image_array = tf.keras.utils.img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)
image_array = image_array / 255.0

# Prediction
prediction = model.predict(image_array, verbose=0)[0]

print("\n==============================")
print("SORGHUM LEAF AI RESULT")
print("==============================")

for i, class_name in enumerate(class_names):
    print(f"{class_name:25}: {prediction[i] * 100:.2f}%")

predicted_index = np.argmax(prediction)

print("\nPrediction:", class_names[predicted_index])
print(f"Confidence: {prediction[predicted_index] * 100:.2f}%")
print("==============================")