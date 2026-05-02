import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from tensorflow.keras.models import load_model
#from keras.models import load_model
# Load model
model = load_model("face_mask_model.h5")
#model = load_model("face_mask_model.keras", compile=False)

st.title("Face Mask Detection App")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -------------------------------
    # Preprocessing (same as training)
    # -------------------------------
    image = image.convert("RGB") ###Added extra
    img = image.resize((225, 225))
    img_array = np.array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # -------------------------------
    # Prediction
    # -------------------------------
    
    prediction = model.predict(img_array, verbose=0) ##Added verbose
    predicted_class = np.argmax(prediction)

    class_labels = ['with_mask', 'without_mask']
    result = class_labels[predicted_class]

    # -------------------------------
    # Output
    # -------------------------------

    if result == "with_mask":
        st.success("Person is wearing a mask 😷")
    else:
        st.error("Person is NOT wearing a mask ⚠️")
