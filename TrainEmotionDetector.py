import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# GPU Optimization
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ GPU detected. Memory growth enabled.")
else:
    print("⚠️ No GPU detected. Running on CPU.")

# Data Paths
train_dir = 'data/train'
validation_dir = 'data/test'
img_size = (128, 128)
batch_size = 32
num_classes = 7  

# Data Augmentation
datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40, 
    width_shift_range=0.2, 
    height_shift_range=0.2,  
    shear_range=0.2,  
    zoom_range=0.3,  
    horizontal_flip=True,
    fill_mode="nearest"
)

datagen_val = ImageDataGenerator(rescale=1./255)

train_set = datagen_train.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
validation_set = datagen_val.flow_from_directory(validation_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

# Compute Class Weights
class_labels = train_set.classes
class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)
class_weights_dict = dict(enumerate(class_weights))

# Model Definition
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze initial layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Optimizer and Callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
model.fit(train_set, validation_data=validation_set, epochs=50, callbacks=[early_stopping, lr_scheduler], class_weight=class_weights_dict)

# Unfreeze some base model layers for fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Re-compile with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training for fine-tuning
model.fit(train_set, validation_data=validation_set, epochs=10, callbacks=[early_stopping, lr_scheduler], class_weight=class_weights_dict)

# Save the model
model.save('model/emotion_model.h5')
print("✅ Model training complete!")
