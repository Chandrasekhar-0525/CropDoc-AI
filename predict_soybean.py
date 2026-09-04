import tensorflow as tf
import json
import numpy as np

model = tf.keras.models.load_model("soybean_disease_model.keras")

with open("soybean_class_names.json", "r") as f:
    class_names = json.load(f)

image_path = input("Enter image path: ")

image = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

image_array = tf.keras.utils.img_to_array(image)
image_array = tf.expand_dims(image_array, 0)

predictions = model.predict(image_array, verbose=0)[0]

predicted_index = np.argmax(predictions)
confidence = predictions[predicted_index] * 100

print("\n==============================")
print("SOYBEAN AI RESULT")
print("==============================")

for i, probability in enumerate(predictions):
    print(f"{class_names[i]:35} : {probability * 100:.2f}%")

print(f"\nPrediction: {class_names[predicted_index]}")
print(f"Confidence: {confidence:.2f}%")
print("==============================")