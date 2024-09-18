from flask import Flask, request, jsonify
import pandas as pd
from prophet import Prophet
import json
import easyocr
import numpy as np
from datetime import datetime, timedelta
import cv2
import mimetypes
import base64
import sys
import spacy
import re

# Load the English NER model from spaCy
nlp = spacy.load("en_core_web_sm")

app = Flask("__main__")


@app.route("/ai/scan", methods=["POST"])
def scan_image():
    try:
        print("Starting image processing...")

        data_bytes = request.get_data()
        data_str = data_bytes.decode('utf-8')
        data_json = json.loads(data_str)
        ext = data_url_to_file(data_json, "./uploads/test")
        file = open("./uploads/test"+ext, 'rb')  # Open the file in binary mode
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        print("Image loaded successfully.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("Image converted to gray.")
        th, threshed = cv2.threshold(gray, 200, 255, cv2.THRESH_TRUNC)
        print("Image threshold applied.")
        
        # Using EasyOCR for text extraction
        reader = easyocr.Reader(['en'])
        result = reader.readtext(threshed)

        # Concatenate text from result
        text = ' '.join([bbox[1] for bbox in result])

        print("Extracted Text:")
        print(text)
        
        output_file = 'extracted_text.txt'
        with open(output_file, 'w', encoding='utf-8') as f:  # Specify encoding as 'utf-8'
            f.write(text)
        print("Text file saved at:", output_file)
        sys.stdout.flush()

        # Extract entities from the text
        entities = extract_entities(text)

        # Write the extracted information to a text file
        with open("extracted_information.txt", "w") as f:
            f.write("Company name: {}\n".format(entities.get('Company name', 'NA')))
            f.write("Contact person name: {}\n".format(entities.get('Contact person name', 'NA')))
            f.write("Email: {}\n".format(entities.get('Email', 'NA')))
            f.write("Mobile number: {}\n".format(entities.get('Mobile number', 'NA')))
            f.write("Address: {}\n".format(entities.get('Address', 'NA')))
            f.write("GST if available: {}\n".format(entities.get('GST if available', 'NA')))

        return jsonify({"text": text, "file_location": output_file, "entities": entities})
    
    except Exception as e:
        print("An error occurred during image processing:", e)
        return jsonify({"error": str(e)})
    
def extract_entities(text):
    entities = {}

    # Extract company name
    company_name_match = re.search(r'[A-Za-z&]+(?:\s+[A-Za-z&]+)*', text)
    if company_name_match:
        entities['Company name'] = company_name_match.group()

    # Extract contact person name
    contact_person_match = re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', text)
    if contact_person_match:
        entities['Contact person name'] = contact_person_match.group()

    # Extract email
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_match:
        entities['Email'] = email_match.group()

    # Extract mobile number
    mobile_number_match = re.search(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
    if mobile_number_match:
        entities['Mobile number'] = mobile_number_match.group()

    # Extract address
    address_match = re.search(r'\d+[\w\s,.]+,\s+\w+[\w\s]+\d+', text)
    if address_match:
        entities['Address'] = address_match.group()

    # Extract GST if available
    gst_match = re.search(r'GST\s*(?:No|Number)[:]?\s*([0-9A-Z]+)', text)
    if gst_match:
        entities['GST if available'] = gst_match.group(1)

    return entities


def data_url_to_file(data_url, output_file_path):
    parts = data_url.split(',')
    if len(parts) != 2:
        raise ValueError("Invalid data URL format")
    mime_type, encoded_data = parts
    extension = mimetypes.guess_extension(mime_type.split(';')[0].split(":")[1])
    if not extension:
        raise ValueError("Could not determine file extension")
    decoded_data = base64.b64decode(encoded_data)
    with open(output_file_path + extension, 'wb') as f:
        f.write(decoded_data)
    return extension

if __name__ == "__main__":
    app.run(debug=True, port=7000)