import gradio as gr
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Global variable to hold the loaded model
model = None

# Fixed path to the model file
MODEL_PATH = "model/model.h5"

def initialize_model():
    """
    Load the Keras model from a fixed path during app initialization.
    """
    global model
    try:
        model = load_model(MODEL_PATH)
        return "Model loaded successfully."
    except Exception as e:
        return f"Failed to load model: {str(e)}"

def process_image(uploaded_image):
    """
    Convert image to four channels, crop it, resize it, and make a prediction.
    """
    if model is None:
        return "No model loaded, please check the server logs for details.", None, None
    
    # Convert PIL image to a numpy array
    img = np.array(uploaded_image)
    
    # Ensure the image has four channels (RGBA)
    if img.shape[2] == 3:  # if the image has 3 channels (RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    
    # Crop the image
    x, y, width, height = 55, 18, 1222 - 55, 620 - 18
    crop_img = img[y:y+height, x:x+width]
    
    # Resize the image
    size = (128, 128)
    resized_img = cv2.resize(crop_img, size)
    
    # Normalize the image
    normalized_img = resized_img / 255.0
    
    # Add batch dimension
    input_img = np.expand_dims(normalized_img, axis=0)
    
    # Make prediction
    prediction = model.predict(input_img)
        #Give the result:
    if(prediction > 0.5):
        result_str = "The image is NOK"
        print("The image is NOK")
    else:
        result_str = "The image is OK"
        print("The image is OK")
    
    # Convert back to PIL image
    result_image = Image.fromarray(resized_img)
    
    return "Image processed successfully.", result_image, result_str       #str(prediction)

# Initialize the model when the script runs
model_status = initialize_model()

# Create the Gradio interface
iface = gr.Interface(
    fn=process_image,  # function to process the image and make prediction
    inputs=gr.Image(type="pil"),  # accept images as PIL type
    outputs=[gr.Textbox(label="Status"), gr.Image(type="pil"), gr.Textbox(label="Prediction")],  # return status, processed image, and prediction
    title="Image Processing and Prediction App",
    description=f"Noise Image Classifier. \n\n **Model Status:** {model_status}"
)

# Run the app
iface.launch()
