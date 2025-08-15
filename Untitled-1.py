pip install tensorflow numpy matplotlib pillow
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


model = VGG16(weights='imagenet')
# Load pre-trained VGG16 model + higher level layers
def load_and_process_image(image_path):
    """
    Load an image from the given path, resize it, and preprocess it for VGG16.
    """
    img = Image.open(image_path)
    img = img.resize((224, 224))  # VGG16 expects 224x224 images
    img_array = np.array(img)
    
    # Ensure the image has 3 color channels (RGB)
    if img_array.shape[-1] != 3:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image for VGG16
    return img_array
#arafat
def predict_image_class(image_path):
    """
    Predict the class of an image using the pre-trained VGG16 model.
    """
    img_array = load_and_process_image(image_path)
    
    # Get predictions
    predictions = model.predict(img_array)
    
    # Decode the top-3 predicted classes
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    
    print("Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i+1}. {label}: {score:.2f}")
    
    # Display the image
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Test with an example image

image_path = 'path_to_your_image.jpg' 
# Replace with the path to your image
predict_image_class(image_path)
