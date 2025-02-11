# Import required packages
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Ensure TensorFlow doesn't allocate all GPU memory
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    try:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU detected. Memory growth set.")
    except RuntimeError as e:
        print(f"⚠️ GPU memory growth error: {e}")

# Disable OpenCL use in OpenCV for compatibility
cv2.ocl.setUseOpenCL(False)

# Define data paths
train_dir = 'data/train'
test_dir = 'data/test'

# Check if directories exist
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError("❌ Check that 'data/train' and 'data/test' directories exist before training.")

print("✅ Data directories found. Proceeding with training.")

# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1.0 / 255)
validation_data_gen = ImageDataGenerator(rescale=1.0 / 255)

# Preprocess all train images
train_generator = train_data_gen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# Preprocess all test images
validation_generator = validation_data_gen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# Create model structure
emotion_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 output classes for emotions
])

# Compile the model
emotion_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001, decay=1e-6),
    metrics=['accuracy']
)

# Train the model
print("🚀 Training started...")
emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)
print("✅ Training completed successfully!")

# Ensure model directory exists
model_dir = "model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save model structure to JSON file
model_json = emotion_model.to_json()
with open(os.path.join(model_dir, "emotion_model.json"), "w") as json_file:
    json_file.write(model_json)

# Save trained model weights
emotion_model.save_weights(os.path.join(model_dir, 'emotion_model.h5'))

print("🎉 Model training completed and saved successfully!")
