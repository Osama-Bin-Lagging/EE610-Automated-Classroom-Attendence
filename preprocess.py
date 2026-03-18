"""
preprocess.py - Dataset preprocessing pipeline
Loads raw student images, fixes orientation, detects faces, crops and saves
normalized face images for training.
"""

import os
import json
import cv2
import numpy as np
from PIL import Image, ImageOps

# Try to import HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False
    print("Warning: pillow-heif not installed. HEIC files will be skipped.")

RAW_DATASET = "course_project_dataset"
PROCESSED_DATASET = "processed_dataset"
FACE_SIZE = 128  # Output face crop size (128x128)
PADDING_RATIO = 0.3  # Extra padding around detected face

# Haar cascade paths (shipped with OpenCV)
CASCADE_DEFAULT = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CASCADE_ALT2 = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"


def load_image_pil(path):
    """Load image with Pillow, handling HEIC and EXIF orientation."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # Fix EXIF orientation
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def pil_to_cv2(pil_img):
    """Convert PIL Image to OpenCV BGR numpy array."""
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def detect_face(gray, cascades):
    """
    Detect the largest face in a grayscale image using Haar cascades.
    Tries multiple cascades and preprocessing if needed.
    Returns (x, y, w, h) or None.
    """
    for cascade_path in cascades:
        cascade = cv2.CascadeClassifier(cascade_path)

        # Try direct detection
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) > 0:
            # Return largest face
            return max(faces, key=lambda f: f[2] * f[3])

        # Try with histogram equalization
        equalized = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(
            equalized, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) > 0:
            return max(faces, key=lambda f: f[2] * f[3])

        # Try with more lenient parameters
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
        )
        if len(faces) > 0:
            return max(faces, key=lambda f: f[2] * f[3])

    return None


def crop_face(img_bgr, face_rect, padding=PADDING_RATIO, output_size=FACE_SIZE):
    """
    Crop face from image with padding, convert to grayscale,
    resize to output_size, and apply CLAHE.
    """
    x, y, w, h = face_rect
    img_h, img_w = img_bgr.shape[:2]

    # Add padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    face_crop = img_bgr[y1:y2, x1:x2]

    # Convert to grayscale
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    # Resize
    gray = cv2.resize(gray, (output_size, output_size))

    # Apply CLAHE for lighting normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return gray


def preprocess_dataset():
    """Process all student images and save face crops."""
    cascades = [CASCADE_DEFAULT, CASCADE_ALT2]

    os.makedirs(PROCESSED_DATASET, exist_ok=True)

    class_list = {}
    stats = {"total": 0, "success": 0, "failed": []}

    student_dirs = sorted(
        d for d in os.listdir(RAW_DATASET)
        if os.path.isdir(os.path.join(RAW_DATASET, d))
    )

    for student_name in student_dirs:
        student_raw_path = os.path.join(RAW_DATASET, student_name)
        student_proc_path = os.path.join(PROCESSED_DATASET, student_name)
        os.makedirs(student_proc_path, exist_ok=True)

        # Get image files
        image_files = sorted(
            f for f in os.listdir(student_raw_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".heif"))
        )

        face_count = 0
        for img_file in image_files:
            stats["total"] += 1
            img_path = os.path.join(student_raw_path, img_file)

            try:
                # Load with Pillow (handles HEIC + EXIF)
                pil_img = load_image_pil(img_path)
                bgr = pil_to_cv2(pil_img)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

                # Detect face
                face_rect = detect_face(gray, cascades)

                if face_rect is not None:
                    face_crop = crop_face(bgr, face_rect)
                    face_count += 1
                    out_path = os.path.join(
                        student_proc_path, f"face_{face_count}.jpg"
                    )
                    cv2.imwrite(out_path, face_crop)
                    stats["success"] += 1
                else:
                    stats["failed"].append(f"{student_name}/{img_file}")
                    print(f"  NO FACE: {student_name}/{img_file}")

            except Exception as e:
                stats["failed"].append(f"{student_name}/{img_file}: {e}")
                print(f"  ERROR: {student_name}/{img_file}: {e}")

        class_list[student_name] = {
            "display_name": student_name,
            "face_count": face_count,
        }
        status = "OK" if face_count >= 3 else "LOW" if face_count > 0 else "FAIL"
        print(f"[{status}] {student_name}: {face_count}/{len(image_files)} faces")

    # Save class list
    with open(os.path.join(PROCESSED_DATASET, "class_list.json"), "w") as f:
        json.dump(class_list, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Preprocessing complete:")
    print(f"  Total images: {stats['total']}")
    print(f"  Faces found:  {stats['success']}")
    print(f"  Failed:       {len(stats['failed'])}")
    if stats["failed"]:
        print(f"\nFailed images:")
        for f in stats["failed"]:
            print(f"  - {f}")

    return stats


if __name__ == "__main__":
    preprocess_dataset()
