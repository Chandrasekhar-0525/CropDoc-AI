import tensorflow as tf
from tensorflow.keras import layers, models
import json

DATASET_PATH = "dataset/sorghum"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load training data
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation data
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

print("\nSorghum classes:")
print(class_names)

# Save class names
with open("sorghum_class_names.json", "w") as f:
    json.dump(class_names, f)

# Create model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\n==============================")
print("TRAINING SORGHUM MODEL")
print("==============================\n")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

model.save("sorghum_disease_model.keras")

print("\n==============================")
print("SORGHUM TRAINING COMPLETE!")
print("==============================")
print("Model saved as sorghum_disease_model.keras")
print("Classes:", class_names)