"""
bench_temporal.py - Multi-frame temporal aggregation benchmark.

Tests strategies for combining predictions across 12 validation images:
baseline (any-1), embedding averaging, confidence-weighted, k-of-n agreement.
Also finds the minimal image subset for maximum recall.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
from face_model import FaceRecognitionModel
from benchmark_detection import load_ground_truth
from cache_detections import load_cache

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")


def eval_strategy(predicted_present, ground_truth):
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
    print("MULTI-FRAME TEMPORAL AGGREGATION BENCHMARK")
    print("=" * 70)

    enrollment_embs, val_detections, image_files = load_cache()

    recognizer = FaceRecognitionModel(threshold=0.05)
    recognizer.load(os.path.join(PROJECT_DIR, "face_database.pkl"))

    gt = load_ground_truth()
    gt_present = {n for n, s in gt.items() if s == "P"}
    n_images = len(image_files)
    print(f"Ground truth: {len(gt_present)} present, {n_images} images")

    # Classify cached detections
    per_student = {}  # {student: [(img_idx, confidence, embedding), ...]}
    per_image = []     # [{label: (conf, emb), ...}, ...]

    for img_idx, img_file in enumerate(image_files):
        img_preds = {}
        for face in val_detections[img_file]:
            label, conf = recognizer.predict(face["embedding"])
            if label != "Unknown":
                per_student.setdefault(label, []).append(
                    (img_idx, float(conf), face["embedding"])
                )
                if label not in img_preds or conf > img_preds[label][0]:
                    img_preds[label] = (float(conf), face["embedding"])
        per_image.append(img_preds)
        print(f"  {img_file}: {len(img_preds)} unique students recognized")

    strategies = {}

    # Baseline (any-1)
    predicted = set(per_student.keys())
    m = eval_strategy(predicted, gt)
    strategies["baseline_any1"] = m
    print(f"\nBaseline (any-1): Recall={m['recall']:.1%} Prec={m['precision']:.1%} F1={m['f1']:.1%}")

    # Embedding average
    avg_predicted = set()
    for student, dets in per_student.items():
        embs = np.array([d[2] for d in dets])
        avg_emb = np.mean(embs, axis=0)
        avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
        label, conf = recognizer.predict(avg_emb)
        if label != "Unknown":
            avg_predicted.add(label)
    m = eval_strategy(avg_predicted, gt)
    strategies["embedding_avg"] = m
    print(f"Embedding Avg: Recall={m['recall']:.1%} Prec={m['precision']:.1%} F1={m['f1']:.1%}")

    # Confidence-weighted sum
    for thresh in [0.05, 0.10, 0.20, 0.50]:
        conf_predicted = set()
        for student, dets in per_student.items():
            if sum(d[1] for d in dets) >= thresh:
                conf_predicted.add(student)
        m = eval_strategy(conf_predicted, gt)
        strategies[f"conf_weighted_t{thresh}"] = m
        print(f"Conf-weighted (t={thresh}): Recall={m['recall']:.1%} Prec={m['precision']:.1%} F1={m['f1']:.1%}")

    # k-of-n agreement
    for k in [1, 2, 3, 4]:
        kn_predicted = set()
        for student, dets in per_student.items():
            if len(set(d[0] for d in dets)) >= k:
                kn_predicted.add(student)
        m = eval_strategy(kn_predicted, gt)
        strategies[f"k_of_n_k{k}"] = m
        print(f"k-of-n (k={k}): {len(kn_predicted)} students → Recall={m['recall']:.1%} "
              f"Prec={m['precision']:.1%} F1={m['f1']:.1%}")

    # Greedy image subset
    print("\n" + "-" * 70)
    print("GREEDY IMAGE SUBSET SELECTION")
    print("-" * 70)

    per_image_students = [set(img_preds.keys()) for img_preds in per_image]
    for img_idx, img_file in enumerate(image_files):
        print(f"  {img_file}: {len(per_image_students[img_idx])} students")

    selected_images = []
    covered = set()
    used = set()

    while len(used) < n_images:
        best_img, best_gain = -1, 0
        for img_idx in range(n_images):
            if img_idx in used:
                continue
            gain = len(per_image_students[img_idx] - covered)
            if gain > best_gain:
                best_gain = gain
                best_img = img_idx
        if best_gain == 0:
            break
        used.add(best_img)
        covered |= per_image_students[best_img]
        recall_so_far = len(covered & gt_present) / len(gt_present) if gt_present else 0
        selected_images.append((best_img, image_files[best_img], best_gain, recall_so_far))
        print(f"  Step {len(selected_images)}: {image_files[best_img]} "
              f"(+{best_gain} new, cumulative recall={recall_so_far:.1%})")

    min_95 = next((i+1 for i, (_, _, _, rec) in enumerate(selected_images) if rec >= 0.95), None)
    print(f"\n  Min images for 95%+ recall: {min_95 if min_95 else '>'+str(n_images)}")

    # Summary
    print(f"\n{'='*80}")
    print("TEMPORAL AGGREGATION RESULTS")
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
        "greedy_image_order": [
            {"step": i+1, "image": fname, "new_students": gain, "cumulative_recall": rec}
            for i, (_, fname, gain, rec) in enumerate(selected_images)
        ],
        "min_images_95_recall": min_95,
        "per_student_detection_count": {
            student: len(set(d[0] for d in dets))
            for student, dets in per_student.items()
        },
    }
    json_path = os.path.join(RESULTS_DIR, "bench_temporal.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
