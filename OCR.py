# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:05:42 2024

@author: 20004806
"""

import cv2
import os
from datetime import datetime
import easyocr
import matplotlib.pyplot as plt
import re
import openpyxl
 
def capture_image_from_camera(ip_address, port, username, password, save_path):
    # URL to access the camera stream
    stream_url = f"rtsp://Username:Password@10.7.IPAddress/Streaming/Channels/1"
 
    # Capture video from the camera stream
    cap = cv2.VideoCapture(stream_url)
 
    # Check if the camera stream is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera stream.")
        return None
    # Get maximum supported resolution
    max_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    max_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # Set properties for better image quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)
    # Read the frame
    ret, frame = cap.read()
 
    # Check if the frame is read successfully
    if not ret:
        print("Error: Unable to read frame.")
        return None
    # Use the current timestamp to generate a unique file name for each image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format the timestamp
    image_filename = f"captured_image_{timestamp}.jpg"  # Create a unique filename
    # Save the captured image
    save_location = os.path.join(save_path, image_filename)
    cv2.imwrite(save_location, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
 
    # Release the capture object and close the stream
    cap.release()
    print("Image captured successfully.")
    return save_location
 
def find_meter_display_area(image, templates):
    best_match = None
    best_val = -1
    best_coords = None
 
    # Iterate through all the templates
    for template_path in templates:
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
        # Template matching
        res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            best_coords = max_loc
            best_match = template
 
    if best_coords is not None:
        h, w = best_match.shape
        top_left = best_coords
        return top_left[0], top_left[1], w, h
    else:
        print("No suitable match found.")
        return None
 
def extract_meter_reading_easyocr(image_path, template_paths):
    # Load the image
    image = cv2.imread(image_path)
 
    # Find the meter display area in the image using multiple templates
    coords = find_meter_display_area(image, template_paths)
 
    if coords is None:
        return "Error: Meter display area not found."
 
    x, y, w, h = coords
    # Crop the image to the region of interest (the meter display)
    cropped_image = image[y:y+h, x:x+w]
    # Convert the cropped image to grayscale
    cropped_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # Resize the image to improve OCR accuracy
    resized = cv2.resize(cropped_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Apply Gaussian blur to the resized grayscale image to reduce noise
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Apply Otsu's thresholding
    _, otsu_thresh = cv2.threshold(adaptive_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])
    # Use EasyOCR to extract text
    result = reader.readtext(otsu_thresh, detail=0)
    # Join the result into a single string
    number_string = ''.join(result)
    # Filter out non-digit characters
    filtered_number_string = re.sub(r'[^0-9]', '', number_string)
    # Display the processed image and the extracted number
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Processed Image')
    plt.imshow(cropped_image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Extracted Number')
    plt.text(0.5, 0.5, filtered_number_string.strip(), fontsize=18, ha='center', va='center')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return filtered_number_string.strip()
 
def save_reading_to_excel(image_name, meter_reading, excel_path):
    # Check if the Excel file exists
    if os.path.exists(excel_path):
        # Load the workbook and select the active sheet
        workbook = openpyxl.load_workbook(excel_path)
        sheet = workbook.active
    else:
        # Create a new workbook and select the active sheet
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        # Write the headers
        sheet.append(["Image Name", "Meter Reading"])
 
    # Append the new data
    sheet.append([image_name, meter_reading])
 
    # Save the workbook
    workbook.save(excel_path)
    print("Meter reading saved to Excel successfully.")
 
# Example usage:
save_path = r"D:\Image Capture"
excel_path = r"D:\Image Capture\Meter Readings.xlsx"
template_paths = [r"D:\Image Capture\Template\Template2.jpg", r"D:\Image Capture\Template\Template.jpg", r"D:\Image Capture\Template\Template3.jpg"]  # List of template paths
 
# Capture the image from the camera
captured_image_path = capture_image_from_camera(ip_address, port, username, password, save_path)
 
if captured_image_path:
    # Extract the meter reading using EasyOCR
    meter_reading = extract_meter_reading_easyocr(captured_image_path, template_paths)
    print("Extracted Meter Reading:", meter_reading)
 
    # Save the reading to the Excel file
    image_name = os.path.basename(captured_image_path)
    save_reading_to_excel(image_name, meter_reading, excel_path)