import tensorflow as tf
import numpy as np
import json
import os

# ==============================
# LOAD MODELS
# ==============================

models = {
    "cotton": (
        "cotton_disease_model.keras",
        "class_names.json"
    ),
    "soybean": (
        "soybean_disease_model.keras",
        "soybean_class_names.json"
    ),
    "sugarcane": (
        "sugarcane_disease_model.keras",
        "sugarcane_class_names.json"
    ),
    "sorghum": (
        "sorghum_disease_model.keras",
        "sorghum_class_names.json"
    )
}

loaded_models = {}

print("\nLoading models...")

for crop, (model_file, class_file) in models.items():

    if not os.path.exists(model_file):
        print(f"❌ {model_file} not found")
        continue

    if not os.path.exists(class_file):
        print(f"❌ {class_file} not found")
        continue

    model = tf.keras.models.load_model(model_file)

    with open(class_file, "r") as f:
        class_names = json.load(f)

    loaded_models[crop] = (model, class_names)

    print(f"✅ {crop} model loaded")

print("\nAll available models loaded.")


# ==============================
# SELECT CROP
# ==============================

print("\n==============================")
print("     CROP DISEASE AI")
print("==============================")

print("1. Cotton")
print("2. Soybean")
print("3. Sugarcane")
print("4. Sorghum")

choice = input("\nSelect crop (1-4): ")

crop_list = {
    "1": "cotton",
    "2": "soybean",
    "3": "sugarcane",
    "4": "sorghum"
}

if choice not in crop_list:
    print("Invalid choice.")
    exit()

crop = crop_list[choice]

if crop not in loaded_models:
    print("Model not available.")
    exit()


# ==============================
# IMAGE PATH
# ==============================

image_path = input("\nEnter image path: ").strip()

if not os.path.exists(image_path):
    print("\n❌ Image not found!")
    print("Example:")
    print("test_images\\test1.jpg")
    exit()


# ==============================
# LOAD IMAGE
# ==============================

model, class_names = loaded_models[crop]

image = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

image_array = tf.keras.utils.img_to_array(image)

image_array = np.expand_dims(image_array, axis=0)

image_array = image_array / 255.0


# ==============================
# PREDICTION
# ==============================

prediction = model.predict(image_array, verbose=0)

predicted_index = np.argmax(prediction[0])

predicted_class = class_names[predicted_index]

confidence = prediction[0][predicted_index] * 100


# ==============================
# RESULT
# ==============================

print("\n==============================")
print("      CROP DISEASE RESULT")
print("==============================")

print(f"Crop       : {crop}")
print(f"Disease    : {predicted_class}")
print(f"Confidence : {confidence:.2f}%")

print("==============================")