"""
recognize.py - Recognition engine
Processes classroom images: detects faces, matches against trained database,
produces annotated images, attendance list, and unknown face crops.
"""

import os
import cv2
import numpy as np
import json
import pandas as pd
from PIL import Image, ImageOps
from lbph import LBPHFaceRecognizer

CASCADE_DEFAULT = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CASCADE_ALT2 = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"

FACE_SIZE = 128
PADDING_RATIO = 0.3
OUTPUT_DIR = "outputs"


def load_image(path):
    """Load image handling EXIF orientation."""
    pil_img = Image.open(path)
    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def detect_faces(gray):
    """Detect all faces in a grayscale image. Returns list of (x, y, w, h)."""
    cascade = cv2.CascadeClassifier(CASCADE_DEFAULT)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        cascade = cv2.CascadeClassifier(CASCADE_ALT2)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
    if len(faces) == 0:
        equalized = cv2.equalizeHist(gray)
        cascade = cv2.CascadeClassifier(CASCADE_DEFAULT)
        faces = cascade.detectMultiScale(
            equalized, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
        )
    return faces


def prepare_face(img_bgr, face_rect):
    """Crop, normalize, and resize a detected face for recognition."""
    x, y, w, h = face_rect
    img_h, img_w = img_bgr.shape[:2]

    pad_w = int(w * PADDING_RATIO)
    pad_h = int(h * PADDING_RATIO)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    face_crop = img_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (FACE_SIZE, FACE_SIZE))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray


def process_classroom_images(image_paths, recognizer, class_list):
    """
    Process classroom images and generate attendance results.

    Returns:
        attendance: dict {student_name: "Present" or "Absent"}
        annotated_images: list of (filename, annotated_bgr_image)
        unknown_faces: list of (face_crop_bgr, distance)
    """
    attendance = {name: "Absent" for name in class_list}
    annotated_images = []
    unknown_faces = []
    all_detections = []

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        bgr = load_image(img_path)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        annotated = bgr.copy()

        faces = detect_faces(gray)
        print(f"  {filename}: {len(faces)} faces detected")

        for (x, y, w, h) in faces:
            face_gray = prepare_face(bgr, (x, y, w, h))
            label, distance = recognizer.predict(face_gray)

            # Store detection
            all_detections.append({
                "file": filename,
                "label": label,
                "distance": distance,
                "bbox": (x, y, w, h),
            })

            if label != "Unknown":
                attendance[label] = "Present"
                color = (0, 200, 0)  # Green for known
                display_name = label.split("(")[0].strip() if "(" in label else label
                # Truncate long names
                if len(display_name) > 20:
                    display_name = display_name[:18] + ".."
            else:
                color = (0, 0, 255)  # Red for unknown
                display_name = "Unknown"
                # Save unknown face crop (color)
                pad_w = int(w * PADDING_RATIO)
                pad_h = int(h * PADDING_RATIO)
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(bgr.shape[1], x + w + pad_w)
                y2 = min(bgr.shape[0], y + h + pad_h)
                unknown_crop = bgr[y1:y2, x1:x2].copy()
                unknown_faces.append((unknown_crop, distance))

            # Draw bounding box and label
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            label_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(
                annotated, display_name, (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        annotated_images.append((filename, annotated))

    return attendance, annotated_images, unknown_faces


def save_outputs(attendance, annotated_images, unknown_faces, class_list):
    """Save all outputs to the outputs directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "annotated"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "unknowns"), exist_ok=True)

    # Save attendance CSV
    rows = []
    for student_name in sorted(class_list.keys()):
        status = attendance.get(student_name, "Absent")
        rows.append({"Name": student_name, "Status": status})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "attendance.csv")
    df.to_csv(csv_path, index=False)
    print(f"Attendance saved to {csv_path}")

    # Save annotated images
    for filename, img in annotated_images:
        out_path = os.path.join(OUTPUT_DIR, "annotated", f"annotated_{filename}")
        cv2.imwrite(out_path, img)
    print(f"Saved {len(annotated_images)} annotated images")

    # Save unknown face crops
    for i, (crop, dist) in enumerate(unknown_faces):
        out_path = os.path.join(OUTPUT_DIR, "unknowns", f"unknown_{i+1}.jpg")
        cv2.imwrite(out_path, crop)
    print(f"Saved {len(unknown_faces)} unknown face crops")

    return csv_path


def train_and_save(processed_dataset="processed_dataset", model_path="face_database.pkl"):
    """Train recognizer and save model."""
    recognizer = LBPHFaceRecognizer(grid_x=8, grid_y=8, threshold=55.0)
    recognizer.train(processed_dataset)
    recognizer.save(model_path)
    return recognizer


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python recognize.py <classroom_image1> [classroom_image2] ...")
        print("  First run: python preprocess.py")
        sys.exit(1)

    model_path = "face_database.pkl"
    class_list_path = os.path.join("processed_dataset", "class_list.json")

    # Load or train model
    recognizer = LBPHFaceRecognizer()
    if os.path.exists(model_path):
        recognizer.load(model_path)
    else:
        print("No model found. Training...")
        recognizer = train_and_save()

    with open(class_list_path) as f:
        class_list = json.load(f)

    image_paths = sys.argv[1:]
    print(f"Processing {len(image_paths)} classroom images...")

    attendance, annotated, unknowns = process_classroom_images(
        image_paths, recognizer, class_list
    )
    save_outputs(attendance, annotated, unknowns, class_list)

    # Print attendance summary
    present = sum(1 for v in attendance.values() if v == "Present")
    absent = sum(1 for v in attendance.values() if v == "Absent")
    print(f"\nAttendance: {present} present, {absent} absent, {len(unknowns)} unknown")
