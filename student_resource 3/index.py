import cv2
import pytesseract
import urllib.request
import ssl
import numpy as np
import pandas as pd
import re
from PIL import Image
import requests
from io import BytesIO

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = '/bin/tesseract'  # Update this path if necessary

# Create an SSL context to ignore SSL verification (not recommended for production)
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Define entity name patterns
ENTITY_MAPPING = {
    'width': 'inch|cm|foot',
    'depth': 'inch|cm|foot',
    'maximum_weight_recommendation': 'kg|ton',
    'voltage': 'volt|V|kilovolt|Kv',
    'wattage': 'watt|kW|kilowatt',
    'item_weight': 'kg|pound|ton',
    'height': 'inch|cm|foot'
}

# Function to extract text from an image URL using OpenCV or PIL
def extract_text_from_image(image_url, use_opencv=True):
    try:
        if use_opencv:
            # Download and read the image using OpenCV
            urllib.request.urlretrieve(image_url, "temp_image.png")
            img = cv2.imread("temp_image.png")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
            contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            extracted_text = ""
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cropped = img[y:y + h, x:x + w]
                text = pytesseract.image_to_string(cropped, config='--psm 6')
                extracted_text += text + "\n"
        else:
            # Using PIL for OCR
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            extracted_text = pytesseract.image_to_string(img)

        return clean_text(extracted_text)

    except Exception as e:
        print(f"Error processing image {image_url}: {str(e)}")
        return ""

# Function to clean extracted text
def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())  # Replace multiple spaces with a single space

# Function to interpret the extracted text and match the entity value
def interpret_text(text, entity_name):
    pattern = ENTITY_MAPPING.get(entity_name.lower(), '')
    if pattern:
        match = re.search(r"(\d+(\.\d+)?)\s*(" + pattern + r")", text, re.IGNORECASE)
        if match:
            return f"{match.group(1)} {match.group(3)}"
    return ""

# Predictor function that integrates text extraction and entity interpretation
def predictor(image_url, group_id, entity_name, use_opencv=True):
    extracted_text = extract_text_from_image(image_url, use_opencv)
    prediction = interpret_text(extracted_text, entity_name)
    return prediction if prediction else "N/A"

# Main execution flow
if __name__ == "__main__":
    # Read test dataset
    input_csv_file = 'dataset/test.csv'
    test_df = pd.read_csv(input_csv_file)

    # Process each row in the dataset
    test_df['prediction'] = test_df.apply(lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)

    # Save the output
    output_filename = 'test_out.csv'
    test_df[['index', 'prediction']].to_csv(output_filename, index=False)

    print(f"Predictions saved to {output_filename}")
