import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model.h5")

def predict(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0]
    
    if prediction[0] > 0.5:
        return "Tumor Detected"
    else:
        return "No Tumor"

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🧠 Brain Tumor Detection",
    description="Upload MRI Image"
).launch()
