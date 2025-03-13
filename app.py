import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('handwritten_digit_model.h5')

# Function to preprocess image and extract number regions
def preprocess_and_extract_numbers(image):
    # Convert PIL image to OpenCV format
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to binarize the image
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Dilate to connect digits within a number (e.g., "12")
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Find contours to detect number regions
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    number_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter small noise and overly large regions
        if 10 < w < 200 and 10 < h < 100:  # Adjust based on your data
            number_regions.append((x, y, w, h))
    
    # Sort regions by x-coordinate (left to right)
    number_regions = sorted(number_regions, key=lambda x: x[0])
    return number_regions, thresh

# Function to recognize a number from a region
def recognize_number(region, thresh):
    x, y, w, h = region
    roi = thresh[y:y+h, x:x+w]
    
    # Estimate number of digits based on width
    digit_width = 28  # Rough estimate for each digit
    num_digits = max(1, w // digit_width)
    
    digits = []
    for i in range(num_digits):
        start_x = i * digit_width
        end_x = min((i + 1) * digit_width, w)
        digit_roi = roi[:, start_x:end_x]
        
        # Pad or resize to 28x28
        digit_roi = cv2.resize(digit_roi, (28, 28), interpolation=cv2.INTER_AREA)
        digit_roi = digit_roi.reshape(1, 28, 28, 1) / 255.0  # Normalize
        
        # Predict the digit
        prediction = model.predict(digit_roi, verbose=0)
        digit = np.argmax(prediction)
        digits.append(digit)
    
    # Combine digits into a single number
    number = int("".join(map(str, digits)))
    return number

# Streamlit app
def main():
    st.title("Handwritten Number Detection and Summation")
    st.write("Upload or capture an image with handwritten numbers (e.g., '1 maggi 12 biscuit 20 biscuit').")
    
    # Option to upload or capture image
    option = st.radio("Choose input method:", ("Upload Image", "Capture from Camera"))
    
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            process_image(image)
    
    elif option == "Capture from Camera":
        camera_image = st.camera_input("Take a picture")
        if camera_image is not None:
            image = Image.open(camera_image)
            st.image(image, caption="Captured Image", use_column_width=True)
            process_image(image)

# Process the image and display results
def process_image(image):
    # Extract number regions
    number_regions, thresh = preprocess_and_extract_numbers(image)
    
    if not number_regions:
        st.write("No numbers detected. Please try a clearer image.")
        return
    
    # Recognize numbers from each region
    numbers = []
    for region in number_regions:
        number = recognize_number(region, thresh)
        numbers.append(number)
    
    st.write("Detected numbers:", numbers)
    
    # Calculate total
    total = sum(numbers)
    st.write(f"Total amount: {total}")

if __name__ == "__main__":
    main()


