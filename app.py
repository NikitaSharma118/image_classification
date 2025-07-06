# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from PIL import Image
# import io
# import json

# # Load your trained model 
# model = tf.keras.models.load_model('model.h5')  # change this to your actual model path
# class_names = ['Lion', 'Tiger', 'Cat', 'Cow','Bear','Bird','Deer','Dog','Dolphin','Elephant','Giraffe','Horse','Kangaroo','Panda','Zebra']  # <-- update with your actual class names

# st.set_page_config(page_title="Image Classifier", layout="centered")
# st.title("🦁 Image Classification Dashboard")

# with open('class_names.json', 'r') as f:
#     class_indices = json.load(f)

# # Upload image
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display image
#     image = Image.open(uploaded_file).convert('RGB')
#     st.image(image, caption='Uploaded Image', use_container_width=True)

#     # Preprocess image
#     img_resized = image.resize((224, 224))
#     img_array = np.array(img_resized) / 255.0  # normalize
#     img_array = np.expand_dims(img_array, axis=0)  # batch dimension

#     # Predict
#     predictions = model.predict(img_array)[0]
#     predicted_index = np.argmax(predictions)
#     predicted_class = class_names[predicted_index]
#     confidence = predictions[predicted_index]

#     # Output
#     st.success(f"Predicted: **{predicted_class}** with **{confidence:.2%}** confidence")

#     # Show bar chart
#     st.subheader("Prediction Confidence for Each Class")
#     st.bar_chart({name: prob for name, prob in zip(class_names, predictions)})

# else:
#     st.info("Please upload an image to get started.")


import streamlit as st
import numpy as np
import json
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

# === Load model and class names ===
model = load_model("model.keras")

with open("class_names.json", "r") as f:
    class_dict = json.load(f)
    class_names = {int(v): k for k, v in class_dict.items()}

# === Preprocessing Function ===
def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)
    return img_array, img

# === Streamlit UI ===
st.set_page_config(page_title="Animal Classifier", layout="centered")
st.title("🦁 Animal Image Classifier")
st.markdown("Upload an image and let the model predict the animal!")

uploaded_image = st.file_uploader("Choose an animal image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    img_array, img_display = preprocess_image(uploaded_image)

    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    st.success(f"**Predicted: {predicted_class}** with **{confidence:.2f}%** confidence")

    # === Plot confidence chart ===
    st.markdown("### Prediction Confidence for Each Class")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(class_names.values(), predictions, color='skyblue')
    ax.set_xticklabels(class_names.values(), rotation=45, ha="right")
    ax.set_ylabel("Confidence")
    st.pyplot(fig)
