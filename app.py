import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("bestmodel_finetuned.keras", compile=False)

# Change classes based on your model
classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

def predict(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0]

    return {
        classes[i]: float(prediction[i]) for i in range(len(classes))
    }

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="🧠 Brain Tumor Detection",
    description="Upload MRI scan to detect tumor type"
)

# IMPORTANT for Render
interface.launch(server_name="0.0.0.0", server_port=7860)
