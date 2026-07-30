import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page setup
st.set_page_config(page_title="🌸 Flower Classifier", page_icon="🌸")
st.title("🌸 Flower Classifier")
st.write("Upload a flower photo and I'll tell you what type of flower it is!")

# Load model
@st.cache_resource
def load_model():
    model = keras.models.load_model("flower_model.h5")
    return model

model = load_model()

# Flower classes
classes = ['Daisy 🌼', 'Dandelion 🌻', 'Rose 🌹', 'Sunflower 🌻', 'Tulip 🌷']

# Upload section
st.subheader("Upload your flower image:")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)
    
    # Predict button
    if st.button("🔍 Classify Flower"):
        img_resized = img.resize((150, 150))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        st.success(f"This is a {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")
        
        # Show all probabilities
        st.subheader("All predictions:")
        for i, cls in enumerate(classes):
            st.write(f"{cls}: {prediction[0][i]*100:.2f}%")