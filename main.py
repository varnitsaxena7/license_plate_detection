import streamlit as st
import cv2
import numpy as np
import easyocr
import pandas as pd
from io import BytesIO
import base64
import requests
from streamlit_lottie import st_lottie
from datetime import datetime

# Load the pre-trained EasyOCR reader
reader = easyocr.Reader(lang_list=['en'])

# Load the Haarcascade XML file for car number plates
plate_cascade = cv2.CascadeClassifier('cars.xml')

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
    
def main():
    st.title("License Plate of Cars Recognition App")
    
    st.sidebar.markdown("<h2 style='color: green;'>Navigation</h2>", unsafe_allow_html=True)
    activities = ["Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 style='color: green;'>Instructions</h3>", unsafe_allow_html=True)
    st.sidebar.info("Select 'Detection' to upload an image and detect license plates. Select 'About' to learn more about the app.")
    
    st.sidebar.markdown("---")
    
    left_column, right_column = st.columns([0.6, 0.4])

    if choice == "Detection":
        with left_column:
            st.markdown("<p style='color: red;'>*Kindly ensure that the image is clear and taken from the correct angles to facilitate accurate license plate detection and reliable results</p>", unsafe_allow_html=True)

            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = read_image(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Detect License Plates"):
                detected_plates = detect_license_plates(image)
                st.write("Detected Plates:")
                st.write(detected_plates)
                
                csv_data = save_to_csv(detected_plates)
                download_link = get_download_link(csv_data)
                st.markdown(download_link, unsafe_allow_html=True)

        with right_column:
            st_lottie(lottie_coding, height="20", key="coding")

    elif choice == "About":
        st.subheader("About the Detection App")

def read_image(uploaded_file):
    image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def detect_license_plates(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
    
    detected_plates = []
    for (x, y, w, h) in plates:
        plate_image = gray[y:y+h, x:x+w]
        texts = reader.readtext(plate_image)
        detected_text = " ".join([text[1] for text in texts])
        detected_plates.append(detected_text)
    
    return detected_plates

def save_to_csv(detected_plates):
    data = []
    for i, plate_text in enumerate(detected_plates):
        data.append({"Index No": i + 1, "License Plate Number": plate_text})
    
    df = pd.DataFrame(data)
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    return csv_data

def get_download_link(data):
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="detected_plates.csv"><button style="background-color: #4CAF50; border: none; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 20px; border-radius: 5px; cursor: pointer;">Download CSV</button></a>'
    return href

if __name__ == "__main__":
    main()
