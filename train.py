import tensorflow as tf
import json

# ==============================
# 1. Settings
# ==============================

DATASET_PATH = "dataset/cotton"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# ==============================
# 2. Load dataset
# ==============================

train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_data.class_names

print("\nClasses found:")
print(class_names)

# Save class names for later prediction
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# Improve data loading speed
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
validation_data = validation_data.prefetch(AUTOTUNE)

# ==============================
# 3. Data augmentation
# ==============================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# ==============================
# 4. Load pretrained AI model
# ==============================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Don't change the pretrained layers initially
base_model.trainable = False

# ==============================
# 5. Build our disease classifier
# ==============================

inputs = tf.keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.2)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

# ==============================
# 6. Compile model
# ==============================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# 7. Train AI
# ==============================

print("\nStarting training...\n")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)

# ==============================
# 8. Save trained model
# ==============================

model.save("cotton_disease_model.keras")

print("\n==============================")
print("TRAINING COMPLETED!")
print("==============================")
print("Model saved as:")
print("cotton_disease_model.keras")

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, "=", name)