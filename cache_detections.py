"""
cache_detections.py - Run all expensive detections once and cache results.

Caches:
1. detect_faces_enhanced() on all 12 val images → per-face bbox/embedding/det_score
2. _extract_embeddings() on enrollment dataset → per-student embedding lists

All bench_*.py scripts load from this cache instead of re-detecting.

Usage: python cache_detections.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
from face_model import FaceRecognitionModel, load_image_rgb
from benchmark_detection import VAL_DIR, DATASET_DIR

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
CACHE_PATH = os.path.join(RESULTS_DIR, "detection_cache.pkl")


def main():
    print("=" * 70)
    print("CACHING DETECTIONS")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model = FaceRecognitionModel(threshold=0.05)

    # 1. Enrollment embeddings
    print("\n[1/2] Extracting enrollment embeddings...")
    enrollment_embs = model._extract_embeddings(DATASET_DIR)
    n_enroll = sum(len(v) for v in enrollment_embs.values())
    print(f"  {len(enrollment_embs)} students, {n_enroll} embeddings")

    # 2. Val image detections
    print("\n[2/2] Running detect_faces_enhanced on val images...")
    image_files = sorted(
        f for f in os.listdir(VAL_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    val_detections = {}  # {filename: [{"bbox": list, "embedding": np.array, "det_score": float}, ...]}
    for img_file in image_files:
        img_path = os.path.join(VAL_DIR, img_file)
        rgb = load_image_rgb(img_path)
        faces = model.detect_faces_enhanced(rgb)

        face_dicts = []
        n_skipped = 0
        for face in faces:
            ds = float(face.det_score) if hasattr(face, "det_score") else 0.0
            if ds < 0.3:
                n_skipped += 1
                continue
            face_dicts.append({
                "bbox": face.bbox.tolist(),
                "embedding": face.embedding.copy(),
                "det_score": ds,
                "kps": face.kps.tolist() if hasattr(face, "kps") and face.kps is not None else None,
            })
        val_detections[img_file] = face_dicts
        skip_str = f" (skipped {n_skipped} low-score)" if n_skipped else ""
        print(f"  {img_file}: {len(face_dicts)} faces{skip_str}")

    total_faces = sum(len(v) for v in val_detections.values())
    print(f"\n  Total: {total_faces} faces across {len(image_files)} images")

    # Save cache
    cache = {
        "enrollment_embeddings": enrollment_embs,
        "val_detections": val_detections,
        "image_files": image_files,
    }
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    size_mb = os.path.getsize(CACHE_PATH) / 1024 / 1024
    print(f"\nCache saved to {CACHE_PATH} ({size_mb:.1f} MB)")


def load_cache():
    """Load the detection cache. Returns (enrollment_embs, val_detections, image_files)."""
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(
            f"Detection cache not found at {CACHE_PATH}. Run: python cache_detections.py"
        )
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    return cache["enrollment_embeddings"], cache["val_detections"], cache["image_files"]


if __name__ == "__main__":
    main()
