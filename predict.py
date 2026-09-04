import tensorflow as tf
import numpy as np
import json

model = tf.keras.models.load_model("cotton_disease_model.keras")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

image_path = input("Enter image path: ")

image = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

image_array = tf.keras.utils.img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)

prediction = model.predict(image_array, verbose=0)[0]

print("\n==============================")
print("COTTON LEAF AI RESULT")
print("==============================")

for i in range(len(class_names)):
    print(f"{class_names[i]:20s}: {prediction[i] * 100:.2f}%")

predicted_index = np.argmax(prediction)

print("\nPrediction:", class_names[predicted_index])
print("Confidence: {:.2f}%".format(
    prediction[predicted_index] * 100
))

print("==============================")