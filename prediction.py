import streamlit as st
from keras.models import load_model
from PIL import Image, UnidentifiedImageError
import numpy as np
from util2 import classify, apply_clahe
import time

# Set title
st.title('Gastrointestinal Disease Classification')

# Set header
st.header('Please upload an endoscopy image')

# Upload file
try:
    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])
except Exception as e:
    st.error("Error loading the image. Please make sure image in jpeg, jpg or png format")

# Load classifier with error handling
try:
    model = load_model("C:/Users/Public/AcademicProject/Training_Model/densenet_2.h5")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Load class names with error handling
try:
    with open('./labels.txt', 'r') as f:
        class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]
except FileNotFoundError:
    st.error("Labels file not found. Please make sure 'labels.txt' is in the correct location.")
    st.stop()
except Exception as e:
    st.error(f"Error reading labels file: {e}")
    st.stop()

# Display image and classify if a file is uploaded
if file is not None:
    try:
        start_time = time.time()
        # Perform CLAHE
        enhanced_image = apply_clahe(file)

        # Load and display the original image
        image = Image.open(file).convert('RGB')
        st.image(image, caption="Original Image", use_column_width=True)
        st.image(enhanced_image, caption="Enhanced Image (CLAHE)", use_column_width=True)
        clahe_time = time.time() - start_time
        print(f"Time for CLAHE processing: {clahe_time:.2f} seconds")

        # Classify the image
        try:
            start_time = time.time()
            class_name, conf_score = classify(enhanced_image, model, class_names)
            st.write("## {}".format(class_name))
            st.write("### Score: {}%".format(int(conf_score * 1000) / 10))
            prediction_time = time.time() - start_time
            print(f"Time for prediction: {prediction_time:.2f} seconds")

        except Exception as e:
            st.error(f"Error during classification: {e}")

    except UnidentifiedImageError:
        st.error("Uploaded file is not a valid image. Please upload a JPEG or PNG image.")
    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")

else:
    st.info("Please upload an image to enhance.")

