"""
recognize.py - Recognition engine
Processes classroom images: detects faces using InsightFace, matches against
trained SVM model, produces annotated images, attendance list, and unknown face crops.
"""

import os
import cv2
import numpy as np
import json
import pandas as pd
from PIL import Image, ImageOps
from face_model import FaceRecognitionModel

OUTPUT_DIR = "outputs"


def load_image(path):
    """Load image handling EXIF orientation. Returns RGB numpy array."""
    pil_img = Image.open(path)
    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return np.array(pil_img)


def process_classroom_images(image_paths, recognizer, class_list):
    """
    Process classroom images and generate attendance results.

    Args:
        image_paths: list of paths to classroom images
        recognizer: trained FaceRecognitionModel instance
        class_list: dict of student names

    Returns:
        attendance: dict {student_name: "Present" or "Absent"}
        annotated_images: list of (filename, annotated_bgr_image)
        unknown_faces: list of (face_crop_bgr, confidence)
    """
    attendance = {name: "Absent" for name in class_list}
    annotated_images = []
    unknown_faces = []

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        rgb = load_image(img_path)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        annotated = bgr.copy()

        # Detect all faces with enhanced detection (RF + Haar union)
        faces = recognizer.detect_faces_enhanced(rgb)
        print(f"  {filename}: {len(faces)} faces detected")

        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            embedding = face.embedding

            # Predict identity
            label, confidence = recognizer.predict(embedding)

            if label != "Unknown":
                attendance[label] = "Present"
                color = (0, 200, 0)  # Green for known (BGR)
                display_name = label.split("(")[0].strip() if "(" in label else label
                if len(display_name) > 20:
                    display_name = display_name[:18] + ".."
                display_text = f"{display_name} ({confidence:.0%})"
            else:
                color = (0, 0, 255)  # Red for unknown (BGR)
                display_text = f"Unknown ({confidence:.0%})"
                # Save unknown face crop
                pad = 10
                cx1 = max(0, x1 - pad)
                cy1 = max(0, y1 - pad)
                cx2 = min(bgr.shape[1], x2 + pad)
                cy2 = min(bgr.shape[0], y2 + pad)
                unknown_crop = bgr[cy1:cy2, cx1:cx2].copy()
                unknown_faces.append((unknown_crop, confidence))

            # Draw bounding box and label
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label_y = y1 - 10 if y1 - 10 > 10 else y2 + 20
            cv2.putText(
                annotated, display_text, (x1, label_y),
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
    for i, (crop, conf) in enumerate(unknown_faces):
        out_path = os.path.join(OUTPUT_DIR, "unknowns", f"unknown_{i+1}.jpg")
        cv2.imwrite(out_path, crop)
    print(f"Saved {len(unknown_faces)} unknown face crops")

    return csv_path


def train_and_save(dataset_path="course_project_dataset", model_path="face_database.pkl"):
    """Train recognizer on raw images and save model."""
    recognizer = FaceRecognitionModel(threshold=0.05)
    recognizer.train(processed_dataset_path=dataset_path)
    recognizer.save(model_path)
    return recognizer


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python recognize.py <classroom_image1> [classroom_image2] ...")
        print("  First run: python preprocess.py  (optional, for face crops)")
        print("  Then:      model trains from raw images in course_project_dataset/")
        sys.exit(1)

    model_path = "face_database.pkl"
    class_list_path = os.path.join("course_project_dataset")

    # Load or train model
    recognizer = FaceRecognitionModel()
    if os.path.exists(model_path):
        recognizer.load(model_path)
    else:
        print("No model found. Training...")
        recognizer = train_and_save()

    # Build class list from dataset folders
    class_list = {
        d: {"display_name": d}
        for d in sorted(os.listdir("course_project_dataset"))
        if os.path.isdir(os.path.join("course_project_dataset", d))
    }

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
