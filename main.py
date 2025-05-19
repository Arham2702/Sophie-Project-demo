import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Facial Feature Scanner", layout="centered")
st.title("🧠 Facial Feature Scanner")

st.write("Upload a clear face image to detect age, gender, and ethnicity.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert to BGR for DeepFace
    img_array = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    with st.spinner("Analyzing face..."):
        try:
            result = DeepFace.analyze(img_bgr, actions=['age', 'gender', 'race'], enforce_detection=True)[0]

            st.success("Face detected and analyzed successfully!")
            st.markdown(f"**Predicted Age:** {int(result['age'])}")
            st.markdown(f"**Predicted Gender:** {result['gender']}")
            st.markdown(f"**Predicted Ethnicity:** {result['dominant_race'].capitalize()}")

        except Exception as e:
            st.error(f"Error: {str(e)}\nPlease upload a clear face image.")

