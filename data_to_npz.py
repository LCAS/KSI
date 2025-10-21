import os
import numpy as np
import cv2
import torch
from pathlib import Path
from models.superpoint import SuperPoint  # Assuming SuperPoint implementation is available
from ultralytics import YOLO
from natsort import natsorted
from tqdm import tqdm

def extract_and_save_features_for_month():
    DESC = 'U-256U-256N-FN-SPBG'
    TYPE = 'long'
    MONTH = 'september'

    # Initialize SuperPoint model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    superpoint_config = {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1
    }
    superpoint = SuperPoint(superpoint_config).to(device)

    # Initialize YOLO model
    yolo = YOLO("./models/weights/yolo.pt").to('cpu')

    input_dir = Path(f'assets/{TYPE}/{MONTH}/rgb/')
    output_dir = Path(f'dump_match_pairs/{DESC}/{MONTH}/npz/')

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files in the input directory, sorted in natural numerical order
    image_extensions = {".jpg", ".png", ".jpeg", ".bmp"}
    image_files = natsorted([p for p in input_dir.rglob("*") if p.suffix.lower() in image_extensions])

    if not image_files:
        print(f"No images found in {input_dir}.")
        return

    for image_path in tqdm(image_files, desc=f"Processing {MONTH}", unit="image"):
        # Read the image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load image: {image_path}")
            continue

        # YOLO processing for masks
        yolo_img = cv2.imread(str(image_path))
        original_height, original_width = yolo_img.shape[:2]
        yolo_img = cv2.resize(yolo_img, (640, 640))
        yolo_result = yolo.predict(yolo_img, conf=0.2, classes=[0, 4], verbose=False)

        if yolo_result[0].masks is not None:
            masks = yolo_result[0].masks.data.cpu().numpy()
            resized_masks = np.empty((masks.shape[0], original_height, original_width), dtype=masks.dtype)
            for j in range(masks.shape[0]):
                resized_masks[j] = cv2.resize(masks[j], (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                resized_masks[j][resized_masks[j] == 1] = 255
                resized_masks[j][resized_masks[j] != 255] = 0
        else:
            resized_masks = None

        # Normalize and convert to tensor
        img_tensor = torch.from_numpy(img / 255.0).float()[None, None].to(device)

        # Extract features
        with torch.no_grad():
            result = superpoint({'image': img_tensor}, resized_masks)

        keypoints = result['keypoints'][0].cpu().numpy()
        descriptors = result['descriptors'][0].cpu().numpy()

        # Save features to npz file in the output directory
        output_file = output_dir / (image_path.name + ".ksi")
        np.savez(output_file, keypoints=keypoints, descriptors=descriptors)

def extract_filenames(input_file, output_file):
    """
    Extract filenames from the input file and write them to the output file.

    Args:
        input_file (str): Path to the input text file.
        output_file (str): Path to the output text file.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Split the line by spaces and extract the first two entries (filenames)
            parts = line.split()
            if len(parts) >= 2:
                outfile.write(f"{parts[0]} {parts[1]}\n")


# Run the function for the specified month
extract_and_save_features_for_month()

# Generate image_pairs_to_match.txt
#extract_filenames('./assets/long/march_test_pairs_gt.txt', 'database_pairs_list.txt')