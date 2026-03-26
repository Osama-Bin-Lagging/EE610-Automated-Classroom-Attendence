"""
bench_quality.py - Face quality-aware filtering benchmark.

Computes quality metrics (det_score, area, sharpness, aspect ratio) for each
detected face and tests quality-based filtering/weighting strategies.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import cv2
import numpy as np
from face_model import FaceRecognitionModel, load_image_rgb
from benchmark_detection import load_ground_truth, VAL_DIR
from cache_detections import load_cache

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")


def compute_sharpness(rgb, bbox):
    """Laplacian variance of face crop as sharpness measure."""
    x1, y1, x2, y2 = [int(c) for c in bbox]
    h, w = rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_composite_quality(face, stats):
    def norm(val, mn, mx):
        return (val - mn) / (mx - mn + 1e-8)
    q_det = norm(face["det_score"], stats["det_min"], stats["det_max"])
    q_area = norm(np.log1p(face["area"]), np.log1p(stats["area_min"]), np.log1p(stats["area_max"]))
    q_sharp = norm(np.log1p(face["sharpness"]), np.log1p(stats["sharp_min"]), np.log1p(stats["sharp_max"]))
    q_aspect = max(0, 1.0 - abs(face["aspect_ratio"] - 1.0))
    return float(np.clip(0.3 * q_det + 0.3 * q_area + 0.3 * q_sharp + 0.1 * q_aspect, 0, 1))


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
    print("FACE QUALITY-AWARE FILTERING BENCHMARK")
    print("=" * 70)

    enrollment_embs, val_detections, image_files = load_cache()

    recognizer = FaceRecognitionModel(threshold=0.05)
    recognizer.load(os.path.join(PROJECT_DIR, "face_database.pkl"))

    gt = load_ground_truth()
    gt_present = {n for n, s in gt.items() if s == "P"}

    # Add quality metrics (sharpness needs the image)
    all_faces = []
    for img_file in image_files:
        img_path = os.path.join(VAL_DIR, img_file)
        rgb = load_image_rgb(img_path)

        for face_dict in val_detections[img_file]:
            bbox = face_dict["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            aspect = (bbox[2] - bbox[0]) / max(bbox[3] - bbox[1], 1)
            sharpness = compute_sharpness(rgb, bbox)

            label, conf = recognizer.predict(face_dict["embedding"])

            all_faces.append({
                "img_file": img_file,
                "label": label,
                "confidence": float(conf),
                "embedding": face_dict["embedding"],
                "bbox": bbox,
                "area": area,
                "aspect_ratio": aspect,
                "sharpness": sharpness,
                "det_score": face_dict["det_score"],
            })
        print(f"  {img_file}: {len(val_detections[img_file])} faces")

    recognized = [f for f in all_faces if f["label"] != "Unknown"]
    print(f"\nTotal faces: {len(all_faces)}, Recognized: {len(recognized)}")

    # Normalization stats
    areas = [f["area"] for f in all_faces]
    sharps = [f["sharpness"] for f in all_faces]
    dets = [f["det_score"] for f in all_faces]
    stats = {
        "area_min": min(areas), "area_max": max(areas),
        "sharp_min": min(sharps), "sharp_max": max(sharps),
        "det_min": min(dets), "det_max": max(dets),
    }

    for face in all_faces:
        face["quality"] = compute_composite_quality(face, stats)

    qualities = [f["quality"] for f in all_faces]
    print(f"\nQuality: min={min(qualities):.3f} median={np.median(qualities):.3f} "
          f"max={max(qualities):.3f} mean={np.mean(qualities):.3f}")

    correct_faces = [f for f in recognized if f["label"] in gt_present]
    incorrect_faces = [f for f in recognized if f["label"] not in gt_present]
    if correct_faces:
        print(f"Correct recognitions avg quality: {np.mean([f['quality'] for f in correct_faces]):.3f}")
    if incorrect_faces:
        print(f"Incorrect recognitions avg quality: {np.mean([f['quality'] for f in incorrect_faces]):.3f}")

    strategies = {}

    # Baseline
    baseline_present = set(f["label"] for f in all_faces if f["label"] != "Unknown")
    m = eval_predictions(baseline_present, gt)
    strategies["baseline"] = m
    print(f"\nBaseline: Recall={m['recall']:.1%} Prec={m['precision']:.1%} F1={m['f1']:.1%}")

    # Quality threshold
    for q_thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        filtered = set(f["label"] for f in all_faces if f["label"] != "Unknown" and f["quality"] >= q_thresh)
        m = eval_predictions(filtered, gt)
        strategies[f"quality_thresh_{q_thresh}"] = m
        print(f"Quality >= {q_thresh}: Recall={m['recall']:.1%} Prec={m['precision']:.1%} F1={m['f1']:.1%} ({len(filtered)} students)")

    # Quality-weighted confidence
    for q_weight in [0.5, 1.0, 2.0]:
        weighted = set()
        for f in all_faces:
            if f["label"] == "Unknown":
                continue
            if f["confidence"] * (f["quality"] ** q_weight) >= recognizer.threshold * 0.5:
                weighted.add(f["label"])
        m = eval_predictions(weighted, gt)
        strategies[f"quality_weighted_w{q_weight}"] = m
        print(f"Quality-weighted (w={q_weight}): Recall={m['recall']:.1%} Prec={m['precision']:.1%} F1={m['f1']:.1%}")

    # Best-quality-per-identity
    best_per_student = {}
    for f in all_faces:
        if f["label"] == "Unknown":
            continue
        s = f["label"]
        if s not in best_per_student or f["quality"] > best_per_student[s]["quality"]:
            best_per_student[s] = f

    best_q_present = set()
    for s, f in best_per_student.items():
        label, conf = recognizer.predict(f["embedding"])
        if label != "Unknown":
            best_q_present.add(label)
    m = eval_predictions(best_q_present, gt)
    strategies["best_quality_per_id"] = m
    print(f"Best-quality-per-ID: Recall={m['recall']:.1%} Prec={m['precision']:.1%} F1={m['f1']:.1%}")

    # Summary
    print(f"\n{'='*80}")
    print("QUALITY FILTERING RESULTS")
    print(f"{'='*80}")
    print(f"{'Strategy':<30} {'TP':>4} {'FP':>4} {'FN':>4} {'Recall':>8} {'Prec':>8} {'F1':>8}")
    print("-" * 70)
    for name, m in strategies.items():
        print(f"{name:<30} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} "
              f"{m['recall']:>7.1%} {m['precision']:>7.1%} {m['f1']:>7.1%}")
    print(f"{'='*80}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_results = {
        "strategies": strategies,
        "quality_stats": {
            "min": float(min(qualities)), "max": float(max(qualities)),
            "mean": float(np.mean(qualities)), "median": float(np.median(qualities)),
        },
        "per_face_quality": [
            {"img_file": f["img_file"], "label": f["label"], "confidence": f["confidence"],
             "quality": f["quality"], "area": f["area"], "sharpness": f["sharpness"],
             "det_score": f["det_score"]}
            for f in all_faces
        ],
    }
    json_path = os.path.join(RESULTS_DIR, "bench_quality.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
