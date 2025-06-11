import os
import tensorflow as tf
import cv2
import numpy as np
import fitz  # PyMuPDF

# Step 1: Suppress oneDNN warnings (optional, based on your logs)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Step 2: Define paths
model_path = "D:/Erp/blank_detector_model.h5"  # Path to the saved model
input_pdf_path = "D:/Erp/answer_sheet.pdf"  # Replace with the path to your input PDF (e.g., "D:/your_directory/input.pdf")
output_pdf_path = "filtered_output.pdf"  # Replace with the desired output PDF path

# Step 3: Load the saved model
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    raise FileNotFoundError(f"Could not load model from {model_path}. Error: {e}. Ensure the model file exists.")

# Step 4: Define preprocessing function (same as used during training)
def preprocess_image(img, target_size=(224, 224)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, target_size)  # Resize to match model input
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Step 5: Define function to classify a page as blank or not
def is_blank_page(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img, verbose=0)[0][0]
    return prediction < 0.5  # Threshold of 0.5 for blank (same as training)

# Step 6: Filter blank pages from the PDF
try:
    # Open the input pdf - - - - - - - >
    input_pdf = fitz.open(input_pdf_path)
    if input_pdf.page_count == 0:
        raise ValueError(f"Input PDF {input_pdf_path} is empty.")

    # Create a new PDF for non-blank pages
    output_pdf = fitz.open()

    # each page needs to be processed seperately - - - - - - - > 
    
    for page_num in range(input_pdf.page_count):
        print(f"Processing page {page_num + 1} of {input_pdf.page_count}...")
        page = input_pdf[page_num]
        # Convert page to image in memory
        pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))  # 150 DPI for speed
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        # Classify the page
        if not is_blank_page(img):
            output_pdf.insert_pdf(input_pdf, from_page=page_num, to_page=page_num)
            print(f"Page {page_num + 1} included (not blank).")
        else:
            print(f"Page {page_num + 1} skipped (blank).")

    # Save the output PDF
    output_pdf.save(output_pdf_path)
    output_pdf.close()
    input_pdf.close()
    print(f"Output PDF saved as {output_pdf_path} with {output_pdf.page_count} pages.")

except Exception as e:
    print(f"An error occurred: {e}")