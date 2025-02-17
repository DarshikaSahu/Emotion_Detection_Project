import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import to_categorical

# Load Model
model = load_model('model/emotion_model.h5')
model.summary()

# Load Validation Dataset (Resize to 128x128 as expected by the model)
validation_set = image_dataset_from_directory(
    "data/test",
    image_size=(128, 128),  
    batch_size=32,
    label_mode="int"  # Load labels as integers
)

# Convert labels to one-hot encoding
def one_hot_labels(dataset, num_classes=7):
    def process(image, label):
        return image, tf.one_hot(label, depth=num_classes)  # Convert to one-hot
    return dataset.map(process)

validation_set = one_hot_labels(validation_set)

# Evaluate Model
def evaluate_model():
    print("Evaluating the model...")
    loss, accuracy = model.evaluate(validation_set)
    print(f'Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

evaluate_model()
