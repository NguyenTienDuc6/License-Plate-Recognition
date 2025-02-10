import os
import cv2
import torch
import csv
import datetime
import function.utils_rotate as utils_rotate
import function.helper as helper

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# License plate detection model
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', 
                                force_reload=True, source='local').to(device)
# License plate OCR model
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local').to(device)
yolo_license_plate.conf = 0.60

# Input folder containing the images
input_folder = "E:/khoa luan/License-Plate-Pics/"

# Output folder for saving annotated images
output_folder = "detected_license_plates"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Path for the CSV file to store license plate data
csv_file_path = os.path.join(output_folder, "license_plate_data.csv")

# Open the CSV file for writing (once, outside the loop)
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write the header row
    csv_writer.writerow(["Image", "License Plate", "Date", "Time"])

    # Loop through all images in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust extensions if needed
            # Full path to the current image
            image_path = os.path.join(input_folder, filename)
            print(f"Processing image: {image_path}")
            
            # Read the image from file
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Could not read image {image_path}. Skipping...")
                continue

            # Convert image to RGB (YoloV5 expects RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect plates in the image
            plates = yolo_LP_detect(rgb_image, size=640)
            
            list_plates = plates.pandas().xyxy[0].values.tolist()

            # Get the file's creation or modification date (using modification date)
            image_modification_time = os.path.getmtime(image_path)
            # Convert to a readable date format
            image_date = datetime.datetime.fromtimestamp(image_modification_time).strftime("%Y-%m-%d")
            
            # Get the current time when processing the image
            current_time = datetime.datetime.now().strftime("%H:%M:%S")

            # Track all license plates for this image
            list_read_plates = set()

            for plate in list_plates:
                x, y, w, h = map(int, [plate[0], plate[1], plate[2] - plate[0], plate[3] - plate[1]])
                crop_img = rgb_image[y:y+h, x:x+w]
                cv2.rectangle(image, (x, y), (x+w, y+h), color=(0,0,225), thickness=2)

                lp = ""
                for cc in range(2):  # Deskewing attempts
                    for ct in range(2):
                        deskewed_img = utils_rotate.deskew(crop_img, cc, ct)
                        lp = helper.read_plate(yolo_license_plate, deskewed_img)
                        if lp != "unknown":
                            list_read_plates.add(lp)
                            cv2.putText(image, lp, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            break
                    if lp != "unknown":
                        break

            # Write all detected license plates from the current image to the CSV file
            for license_plate in list_read_plates:
                csv_writer.writerow([filename, license_plate, image_date, current_time])

            # Save the annotated image with detected plates and labels
            annotated_image_path = os.path.join(output_folder, f"annotated_{filename}")
            cv2.imwrite(annotated_image_path, image)

            print(f"Processed and saved: {annotated_image_path}")

print("Processing complete.")
