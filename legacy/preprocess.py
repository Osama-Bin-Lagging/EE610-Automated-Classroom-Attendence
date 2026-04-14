"""
preprocess.py - Dataset preprocessing pipeline
Loads raw student images, fixes orientation, detects faces using RetinaFace
(via InsightFace), crops and saves normalized RGB face images for training.
Optionally generates augmented images.
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
FACE_SIZE = 112  # InsightFace expects 112x112 for ArcFace
PADDING_RATIO = 0.2  # Extra padding around detected face

# Lazy-loaded InsightFace app
_face_app = None


def _get_face_app():
    """Lazy-initialize InsightFace detector."""
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


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


def detect_face_insightface(rgb_image):
    """
    Detect the largest face in an RGB image using RetinaFace.
    Returns (x1, y1, x2, y2) bounding box or None.
    """
    app = _get_face_app()
    faces = app.get(rgb_image)
    if not faces:
        return None
    # Return largest face bbox
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.bbox.astype(int)  # [x1, y1, x2, y2]


def crop_face_rgb(rgb_image, bbox, padding=PADDING_RATIO, output_size=FACE_SIZE):
    """
    Crop face from RGB image with padding, resize to output_size.
    Returns RGB numpy array.
    """
    x1, y1, x2, y2 = bbox
    h, w = rgb_image.shape[:2]
    face_w = x2 - x1
    face_h = y2 - y1

    # Add padding
    pad_w = int(face_w * padding)
    pad_h = int(face_h * padding)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    face_crop = rgb_image[y1:y2, x1:x2]

    # Resize to output_size
    face_crop = cv2.resize(face_crop, (output_size, output_size))

    return face_crop


def preprocess_dataset(use_augmentation=False, n_augmented=2):
    """
    Process all student images and save face crops.
    If use_augmentation=True, also generates augmented versions.
    """
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
                rgb = np.array(pil_img)

                # Detect face with RetinaFace
                bbox = detect_face_insightface(rgb)

                if bbox is not None:
                    face_crop = crop_face_rgb(rgb, bbox)
                    face_count += 1
                    out_path = os.path.join(
                        student_proc_path, f"face_{face_count}.jpg"
                    )
                    # Save as RGB (convert to BGR for cv2.imwrite)
                    cv2.imwrite(out_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
                    stats["success"] += 1

                    # Generate augmented versions
                    if use_augmentation:
                        from augment import generate_augmented_images
                        augmented = generate_augmented_images(
                            Image.fromarray(face_crop), n_augmented=n_augmented
                        )
                        for aug_img in augmented:
                            face_count += 1
                            aug_path = os.path.join(
                                student_proc_path, f"face_{face_count}.jpg"
                            )
                            aug_rgb = np.array(aug_img)
                            cv2.imwrite(aug_path, cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR))
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
