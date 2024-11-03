import streamlit as st
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

# Set the path to the Tesseract executable if needed
# Uncomment and set the path if Tesseract is not in your PATH
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_tesseract>'

def preprocess_image(image):
    """ Preprocess the image for better OCR results. """
    # Convert to grayscale
    gray_image = image.convert("L")
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(gray_image)
    enhanced_image = enhancer.enhance(2)

    # Apply thresholding (convert to binary)
    bw_image = enhanced_image.point(lambda x: 0 if x < 128 else 255, '1')

    # Optional: Apply a slight blur to reduce noise
    denoised_image = bw_image.filter(ImageFilter.MedianFilter(size=3))

    # Resize the image for better results using LANCZOS filter
    resized_image = denoised_image.resize((denoised_image.width * 2, denoised_image.height * 2), Image.LANCZOS)

    return resized_image

# Streamlit app
st.title("Image Text Extraction Using OCR")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using PIL
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Display the processed image for reference
    st.image(processed_image, caption="Processed Image", use_column_width=True)

    # Extract text from the processed image
    custom_config = r'--oem 3 --psm 6 -l eng'  # Specify English language
    extracted_text = pytesseract.image_to_string(processed_image, config=custom_config)

    # Display the extracted text
    st.subheader("Extracted Text:")
    st.write(extracted_text)
