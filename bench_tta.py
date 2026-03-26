"""
bench_tta.py - Test-Time Augmentation benchmark.

For each detected face, creates multiple augmented versions (flip, rotate, brightness),
extracts embeddings for each, averages them, and reclassifies. Compares vs baseline.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import cv2
import numpy as np
from PIL import Image
from face_model import FaceRecognitionModel, load_image_rgb, _get_face_app
from benchmark_detection import load_ground_truth, VAL_DIR
from cache_detections import load_cache

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")


def get_embedding_from_crop(crop_rgb):
    """Run InsightFace on a crop to get embedding."""
    app = _get_face_app()
    faces = app.get(crop_rgb)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.embedding


def generate_tta_versions(crop_rgb):
    """Generate TTA versions of a face crop. Yield one at a time to save memory."""
    yield crop_rgb
    yield np.fliplr(crop_rgb).copy()
    yield np.clip(crop_rgb.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)


def eval_predictions(predicted_present, ground_truth):
    gt_present = {n for n, s in ground_truth.items() if s == "P"}
    gt_absent = {n for n, s in ground_truth.items() if s == "A"}
    tp = len(predicted_present & gt_present)
    fp = len(predicted_present & gt_absent)
    fn = len(gt_present - predicted_present)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def main():
    print("=" * 70)
    print("TEST-TIME AUGMENTATION BENCHMARK")
    print("=" * 70)

    _, val_detections, image_files = load_cache()

    recognizer = FaceRecognitionModel(threshold=0.05)
    recognizer.load(os.path.join(PROJECT_DIR, "face_database.pkl"))
    gt = load_ground_truth()

    baseline_present = set()
    tta_present = set()
    per_face_results = []
    total_faces = 0
    tta_success = 0

    import gc
    for img_file in image_files:
        img_path = os.path.join(VAL_DIR, img_file)
        rgb = load_image_rgb(img_path)
        cached_faces = val_detections[img_file]
        print(f"\n  {img_file}: {len(cached_faces)} faces", flush=True)

        for face_dict in cached_faces:
            total_faces += 1
            emb = face_dict["embedding"]
            bbox = face_dict["bbox"]

            # Baseline
            baseline_label, baseline_conf = recognizer.predict(emb)
            if baseline_label != "Unknown":
                baseline_present.add(baseline_label)

            # TTA: crop face from image, augment, re-embed, average
            h, w = rgb.shape[:2]
            x1, y1, x2, y2 = [int(c) for c in bbox]
            bw, bh = x2 - x1, y2 - y1
            pad = int(max(bw, bh) * 0.3)
            cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
            cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
            crop = rgb[cy1:cy2, cx1:cx2]

            if crop.size == 0:
                if baseline_label != "Unknown":
                    tta_present.add(baseline_label)
                per_face_results.append({
                    "img_file": img_file, "baseline_label": baseline_label,
                    "baseline_conf": float(baseline_conf),
                    "tta_label": baseline_label, "tta_conf": float(baseline_conf),
                    "tta_versions_used": 0,
                })
                continue

            tta_embeddings = []
            for version in generate_tta_versions(crop):
                e = get_embedding_from_crop(version)
                if e is not None:
                    tta_embeddings.append(e / (np.linalg.norm(e) + 1e-8))
                del version

            if tta_embeddings:
                tta_success += 1
                avg_emb = np.mean(tta_embeddings, axis=0)
                avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
                tta_label, tta_conf = recognizer.predict(avg_emb)
            else:
                tta_label, tta_conf = baseline_label, baseline_conf

            if tta_label != "Unknown":
                tta_present.add(tta_label)

            per_face_results.append({
                "img_file": img_file,
                "baseline_label": baseline_label, "baseline_conf": float(baseline_conf),
                "tta_label": tta_label, "tta_conf": float(tta_conf),
                "tta_versions_used": len(tta_embeddings),
                "conf_delta": float(tta_conf - baseline_conf),
            })

        del rgb
        gc.collect()

    baseline_metrics = eval_predictions(baseline_present, gt)
    tta_metrics = eval_predictions(tta_present, gt)
    tta_rate = tta_success / total_faces if total_faces > 0 else 0
    conf_deltas = [r.get("conf_delta", 0) for r in per_face_results if "conf_delta" in r]
    avg_conf_delta = np.mean(conf_deltas) if conf_deltas else 0
    label_changes = sum(1 for r in per_face_results if r["baseline_label"] != r["tta_label"])

    print(f"\n{'='*70}")
    print("TEST-TIME AUGMENTATION RESULTS")
    print(f"{'='*70}")
    print(f"  Total faces: {total_faces}, TTA success: {tta_rate:.1%}")
    print(f"  Avg confidence delta: {avg_conf_delta:+.4f}, Label changes: {label_changes}")
    print(f"\n  {'Method':<15} {'TP':>4} {'FP':>4} {'FN':>4} {'Recall':>8} {'Prec':>8} {'F1':>8}")
    print("  " + "-" * 55)
    for name, m in [("Baseline", baseline_metrics), ("TTA", tta_metrics)]:
        print(f"  {name:<15} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} "
              f"{m['recall']:>7.1%} {m['precision']:>7.1%} {m['f1']:>7.1%}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_results = {
        "baseline": baseline_metrics, "tta": tta_metrics,
        "total_faces": total_faces, "tta_success_rate": float(tta_rate),
        "avg_conf_delta": float(avg_conf_delta), "label_changes": label_changes,
        "per_face": per_face_results,
    }
    json_path = os.path.join(RESULTS_DIR, "bench_tta.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
