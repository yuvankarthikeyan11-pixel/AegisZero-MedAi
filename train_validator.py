import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ========================
# PATHS
# ========================
DATASET_DIR = "xray_dataset"
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10

print("🚀 Training Chest X-ray Validator...")

# ========================
# DATA GENERATOR
# ========================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# ========================
# LOAD BASE MODEL
# ========================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

# ========================
# CUSTOM HEAD
# ========================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ========================
# TRAIN
# ========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ========================
# SAVE MODEL
# ========================
model.save("models/xray_validator.h5")

print("\n✅ Validator training complete!")