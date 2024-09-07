import streamlit as st
import numpy as np
from PIL import Image
from keras._tf_keras.keras.models import load_model

# Load fraud detection models
model_500rs = load_model('cnndikbhai.h5')
model_1000rs = load_model('xx1000.h5')

# Function to preprocess the image before feeding it to the detection model
def preprocess_detection_image(image):
    try:
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        # If the image is grayscale or has an alpha channel, adjust accordingly
        # if image.shape[-1] == 1:
        #     image = np.repeat(image, 3, axis=-1)
        # elif image.shape[-1] == 4:
        #     image = image[..., :3]
        # Reshape to (1,224,224,3)
        image = image.reshape((1, 224, 224, 3))
        return image
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to detect fraud using the selected model
def detect_fraud(img, model):
    if img is None:
        raise ValueError("Invalid image input")
    prediction = model.predict(img)
    print(prediction)
    # Convert prediction to binary classification
    prediction = prediction > 0.5
    label = int(prediction[0][0])
    print(label)
    if label == 0:
        return 'Fake'
    elif label == 1:
        return 'Genuine'
    else:
        return 'Unknown'

# Custom CSS for sticky navigation bar
st.markdown("""
    <style>
    .navbar {
        display: flex;
        justify-content: center;
        background-color: #343a40;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .navbar h1 {
        margin: 0;
        font-size: 1.5rem;
        font-family: Monospace;
        color: white;
    }
    .content {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 20px;
    }
    .image-preview {
        margin-bottom: 20px;
    }
    .analyze-button {
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        border-top: 1px solid #e1e1e1;
        padding: 10px;
    }
    </style>
    <div class="navbar">
        <h1>💵 Fake Nepali Currency Detection<h1 style='font-size:30px;'>🇳🇵</h1></h1>
    </div>
    """, unsafe_allow_html=True)

# Main content
st.markdown("<div class='content'>", unsafe_allow_html=True)
st.write("### Upload image for detection:")

# Currency selection
currency_type = st.selectbox("Select the currency type:", ["500 Rupees", "1000 Rupees"])

# Set model based on selected currency
if currency_type == "500 Rupees":
    selected_model = model_500rs
elif currency_type == "1000 Rupees":
    selected_model = model_1000rs

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"], help="Upload a JPG or JPEG image of the currency note.")

if uploaded_file is not None:
    st.markdown("<div class='image-preview'>", unsafe_allow_html=True)
    st.image(uploaded_file, caption="Uploaded Image", width=100)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='analyze-button'>", unsafe_allow_html=True)
    analyze_button = st.button("Analyze Image")
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze_button:
        with st.spinner('Analyzing the image...'):
            fraud_detection_img = preprocess_detection_image(Image.open(uploaded_file))
            if fraud_detection_img is not None:
                fraud_result = detect_fraud(fraud_detection_img, selected_model)
                if fraud_result == 'Genuine':
                    st.success(f"🎉 The currency note is: **{fraud_result}**")
                    with open('real.html', 'r', encoding="utf-8") as file:
                        html_content = file.read()
                    st.components.v1.html(html_content, height=100)
                else:
                    st.error(f"🚨 The currency note is: **{fraud_result}**")
                    with open('fake.html', 'r', encoding="utf-8") as file:
                        html_content = file.read()
                    st.components.v1.html(html_content, height=100)
            else:
                st.error("Image preprocessing failed. Cannot proceed with fraud detection.")
else:
    st.info("Please upload an image file to proceed.")

# Close the content div
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>© 2024 Fake Nepali Currency Detection</p>
    </div>
    """, unsafe_allow_html=True)
