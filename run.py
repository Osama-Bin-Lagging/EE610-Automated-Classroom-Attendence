"""
run.py - Production attendance pipeline.

Usage:
    python run.py val_data/*.JPG              # run attendance on images
    python run.py --train                     # retrain augmented model
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import gc
import json
import csv
import pickle
import argparse
import numpy as np
import cv2
from PIL import Image, ImageOps

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "course_project_dataset")
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
MODEL_PATH = os.path.join(PROJECT_DIR, "face_database_aug.pkl")
AUG_CACHE = os.path.join(RESULTS_DIR, "aug_cache_n2.pkl")

THRESHOLD = 0.03
DET_SCORE_MIN = 0.3
N_AUG = 2


def train_model():
    """Train SVM with 2x augmentation, save to face_database_aug.pkl."""
    from face_model import FaceRecognitionModel, load_image_rgb
    from augment import generate_augmented_images

    model = FaceRecognitionModel(threshold=THRESHOLD)

    if os.path.exists(AUG_CACHE):
        print(f"Loading augmented embeddings from cache: {AUG_CACHE}")
        with open(AUG_CACHE, "rb") as f:
            cached = pickle.load(f)
        embeddings_dict = cached["embeddings_dict"]
    else:
        print("Extracting embeddings with 2x augmentation...")
        # Get base enrollment embeddings
        embeddings_dict = model._extract_embeddings(DATASET_DIR)

        # Add augmented embeddings
        student_dirs = sorted(
            d for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        )
        attempted, success = 0, 0
        for name in student_dirs:
            embs = list(embeddings_dict.get(name, []))
            student_path = os.path.join(DATASET_DIR, name)
            image_fnames = sorted(
                fn for fn in os.listdir(student_path)
                if fn.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".heif"))
            )
            for fn in image_fnames:
                try:
                    pil_img = Image.open(os.path.join(student_path, fn))
                    pil_img = ImageOps.exif_transpose(pil_img)
                    if pil_img.mode != "RGB":
                        pil_img = pil_img.convert("RGB")
                except Exception:
                    continue
                aug_imgs = generate_augmented_images(pil_img, N_AUG)
                del pil_img
                for aug_img in aug_imgs:
                    attempted += 1
                    aug_emb = model.get_embedding(np.array(aug_img))
                    if aug_emb is not None:
                        embs.append(aug_emb)
                        success += 1
                del aug_imgs
            gc.collect()
            if embs:
                embeddings_dict[name] = embs
            print(f"  {name}: {len(embs)} embeddings")

        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(AUG_CACHE, "wb") as f:
            pickle.dump({"embeddings_dict": embeddings_dict,
                         "attempted": attempted, "success": success}, f)
        print(f"Saved augmentation cache: {AUG_CACHE}")

    model.train(embeddings_dict=embeddings_dict)
    model.save(MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")
    return model


def load_model():
    """Load trained model, training if needed."""
    from face_model import FaceRecognitionModel
    model = FaceRecognitionModel(threshold=THRESHOLD)
    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Training...")
        return train_model()
    model.load(MODEL_PATH)
    model.threshold = THRESHOLD
    return model


def next_run_dir():
    """Return path to next runs/run_N directory."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    existing = [
        d for d in os.listdir(RUNS_DIR)
        if d.startswith("run_") and os.path.isdir(os.path.join(RUNS_DIR, d))
    ]
    nums = []
    for d in existing:
        try:
            nums.append(int(d.split("_", 1)[1]))
        except ValueError:
            pass
    n = max(nums, default=0) + 1
    return os.path.join(RUNS_DIR, f"run_{n}")


def detect_image(model, img_path):
    """Run enhanced detection on one image. Returns list of face dicts."""
    from face_model import load_image_rgb
    rgb = load_image_rgb(img_path)
    faces = model.detect_faces_enhanced(rgb)
    results = []
    for face in faces:
        ds = float(face.det_score) if hasattr(face, "det_score") else 0.0
        if ds < DET_SCORE_MIN:
            continue
        results.append({
            "bbox": face.bbox.tolist(),
            "embedding": face.embedding.copy(),
            "det_score": ds,
            "kps": face.kps.tolist() if hasattr(face, "kps") and face.kps is not None else None,
        })
    return results, rgb


def draw_annotated(rgb, faces_in_image, out_path):
    """Draw bboxes on image and save."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for f in faces_in_image:
        x1, y1, x2, y2 = [int(c) for c in f["bbox"]]
        label = f.get("label", "Unknown")
        conf = f.get("confidence", 0)
        det = f.get("det_score", 0)

        if label.startswith("Unknown"):
            color = (0, 0, 220)   # red
            text = f"? (d={det:.2f})"
        else:
            color = (0, 200, 0)   # green
            text = label

        # Scale text to face size
        face_w = x2 - x1
        scale = max(0.3, face_w / 200.0)
        thickness = max(1, int(scale * 2))

        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, max(1, int(scale * 2)))
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(bgr, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(bgr, text, (x1, y1 - 2), font, scale, (255, 255, 255), thickness)
    cv2.imwrite(out_path, bgr)


def save_crop(rgb, bbox, out_path):
    """Save a face crop from RGB image."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = [max(0, int(c)) for c in bbox]
    x2, y2 = min(w, x2), min(h, y2)
    crop = rgb[y1:y2, x1:x2]
    if crop.size > 0:
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, bgr)


def run_pipeline(image_paths):
    """Main pipeline: detect → re-ID → output."""
    from reid import PersonReIdentifier

    model = load_model()
    run_dir = next_run_dir()
    ann_dir = os.path.join(run_dir, "annotated")
    unk_dir = os.path.join(run_dir, "unknowns")
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(unk_dir, exist_ok=True)
    print(f"\nOutput: {run_dir}")

    # Phase 1: Detection
    print("\n[1/3] Detecting faces...")
    detections_by_image = {}
    rgb_images = {}
    image_files = []
    for path in image_paths:
        fname = os.path.basename(path)
        image_files.append(fname)
        dets, rgb = detect_image(model, path)
        detections_by_image[fname] = dets
        rgb_images[fname] = rgb
        print(f"  {fname}: {len(dets)} faces")

    total_faces = sum(len(v) for v in detections_by_image.values())
    print(f"  Total: {total_faces} faces across {len(image_files)} images")

    # Phase 2: Re-ID
    print("\n[2/3] Running re-identification...")
    reid = PersonReIdentifier(merge_threshold=0.6)
    person_sets = reid.process_all(
        detections_by_image, model.predict, image_files
    )
    attendance = reid.get_attendance()
    reid_results = reid.get_results()

    n_labeled = reid_results["stats"]["labeled"]
    n_unknown = reid_results["stats"]["unlabeled"]
    print(f"  {n_labeled} recognized, {n_unknown} unknown person groups")

    # Phase 3: Outputs
    print("\n[3/3] Generating outputs...")

    # Build per-image face annotation data from person sets
    ann_by_image = {f: [] for f in image_files}
    for ps in person_sets:
        for face in ps.faces:
            ann_by_image[face.image_file].append({
                "bbox": [float(c) for c in face.bbox],
                "label": ps.label,
                "confidence": float(face.svm_confidence),
                "det_score": float(face.det_score),
            })

    # Annotated images
    for fname in image_files:
        draw_annotated(rgb_images[fname], ann_by_image[fname],
                       os.path.join(ann_dir, fname))

    # Unknown crops grouped by person set
    for ps in person_sets:
        if not ps.label.startswith("Unknown Person"):
            continue
        person_dir = os.path.join(unk_dir, ps.label.replace(" ", "_").replace("#", ""))
        os.makedirs(person_dir, exist_ok=True)
        for i, face in enumerate(ps.faces):
            crop_name = f"{face.image_file.rsplit('.', 1)[0]}_face{i}.jpg"
            save_crop(rgb_images[face.image_file], face.bbox,
                      os.path.join(person_dir, crop_name))

    # attendance.csv
    all_students = sorted(model.labels)
    csv_path = os.path.join(run_dir, "attendance.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Status", "Confidence", "Images_Detected_In"])
        for name in all_students:
            if name in attendance:
                a = attendance[name]
                writer.writerow([
                    name, "Present",
                    f"{a['max_confidence']:.4f}",
                    a["images_detected_in"],
                ])
            else:
                writer.writerow([name, "Absent", "", ""])

    n_present = sum(1 for n in all_students if n in attendance)
    print(f"  attendance.csv: {n_present}/{len(all_students)} present")
    print(f"  annotated/: {len(image_files)} images")

    # summary.json
    summary = {
        "config": {
            "threshold": THRESHOLD,
            "det_score_min": DET_SCORE_MIN,
            "n_augmentation": N_AUG,
            "model_path": MODEL_PATH,
            "n_enrolled_students": len(model.labels),
        },
        "input_images": image_files,
        "totals": {
            "faces_detected": total_faces,
            "students_present": n_present,
            "students_absent": len(all_students) - n_present,
            "unknown_person_groups": n_unknown,
        },
        "per_image": {
            fname: {
                "faces_detected": len(detections_by_image[fname]),
                "recognized": sum(
                    1 for a in ann_by_image[fname]
                    if not a["label"].startswith("Unknown")
                ),
            }
            for fname in image_files
        },
        "reid_stats": reid_results["stats"],
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results in {run_dir}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Classroom attendance pipeline")
    parser.add_argument("images", nargs="*", help="Input image paths")
    parser.add_argument("--train", action="store_true",
                        help="Retrain model with 2x augmentation")
    args = parser.parse_args()

    if args.train:
        train_model()
        return

    if not args.images:
        parser.error("Provide image paths or use --train")

    # Verify all images exist
    for p in args.images:
        if not os.path.isfile(p):
            print(f"Error: {p} not found")
            sys.exit(1)

    run_pipeline(args.images)


if __name__ == "__main__":
    main()
