"""
bench_svm_tuning.py - Test three strategies to recover missed students.

Strategy 1: Threshold sweep (0.01 to 0.05)
Strategy 2: Per-student adaptive threshold based on enrollment embedding spread
Strategy 3: Cosine-similarity fallback when SVM says Unknown

Evaluates each on cached val detections vs ground truth.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from face_model import FaceRecognitionModel
from cache_detections import load_cache
from benchmark_detection import load_ground_truth

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")


def evaluate(predicted_present, ground_truth):
    gt_present = {n for n, s in ground_truth.items() if s == "P"}
    gt_absent = {n for n, s in ground_truth.items() if s == "A"}
    tp = len(predicted_present & gt_present)
    fp = len(predicted_present & gt_absent)
    fn = len(gt_present - predicted_present)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    missed = sorted(gt_present - predicted_present)
    false_pos = sorted(predicted_present & gt_absent)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
            "missed": missed, "false_positives": false_pos}


def strategy_threshold_sweep(enrollment_embs, val_detections, image_files, gt):
    """Strategy 1: Try different global thresholds."""
    print("\n" + "=" * 70)
    print("STRATEGY 1: GLOBAL THRESHOLD SWEEP")
    print("=" * 70)

    thresholds = [0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    results = []

    for thresh in thresholds:
        model = FaceRecognitionModel(threshold=thresh)
        model.train(embeddings_dict=enrollment_embs)

        predicted = set()
        for img_file in image_files:
            for face in val_detections[img_file]:
                label, conf = model.predict(face["embedding"])
                if label != "Unknown":
                    predicted.add(label)

        m = evaluate(predicted, gt)
        results.append({"threshold": thresh, **m})
        print(f"  thresh={thresh:.3f}  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  "
              f"Prec={m['precision']:.1%}  Rec={m['recall']:.1%}  F1={m['f1']:.1%}  "
              f"Missed={m['missed']}")

    return results


def strategy_adaptive_threshold(enrollment_embs, val_detections, image_files, gt):
    """Strategy 2: Per-student threshold based on enrollment embedding compactness."""
    print("\n" + "=" * 70)
    print("STRATEGY 2: PER-STUDENT ADAPTIVE THRESHOLD")
    print("=" * 70)

    # Train base model
    model = FaceRecognitionModel(threshold=0.05)
    model.train(embeddings_dict=enrollment_embs)

    # Compute per-student centroid and spread (avg pairwise cosine distance)
    centroids = {}
    spreads = {}
    for name, embs in enrollment_embs.items():
        embs_arr = np.array(embs)
        centroid = embs_arr.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        centroids[name] = centroid

        # Average cosine sim of embeddings to centroid
        norms = embs_arr / (np.linalg.norm(embs_arr, axis=1, keepdims=True) + 1e-10)
        sims = norms @ centroid
        spreads[name] = float(np.mean(sims))

    # Students with tighter clusters (higher avg sim) are more reliable
    # → can use lower threshold. Loose clusters need higher threshold.
    # Map: spread in [min, max] → threshold in [0.02, 0.06]
    spread_vals = np.array(list(spreads.values()))
    s_min, s_max = spread_vals.min(), spread_vals.max()

    multipliers = [0.5, 0.6, 0.7, 0.8]
    results = []

    for mult in multipliers:
        # For each student: threshold = base * (1 - mult * normalized_compactness)
        # More compact → lower threshold
        base_thresh = 0.05
        student_thresholds = {}
        for name in enrollment_embs:
            compactness = (spreads[name] - s_min) / (s_max - s_min + 1e-10)
            student_thresholds[name] = base_thresh * (1 - mult * compactness)

        predicted = set()
        for img_file in image_files:
            for face in val_detections[img_file]:
                probs = model.svm.predict_proba(face["embedding"].reshape(1, -1))[0]
                max_idx = np.argmax(probs)
                conf = probs[max_idx]
                label = model.label_encoder.inverse_transform([max_idx])[0]
                if conf >= student_thresholds[label]:
                    predicted.add(label)

        m = evaluate(predicted, gt)
        results.append({"multiplier": mult, **m})
        print(f"  mult={mult:.1f}  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  "
              f"Prec={m['precision']:.1%}  Rec={m['recall']:.1%}  F1={m['f1']:.1%}  "
              f"Missed={m['missed']}")

    # Show per-student thresholds for best multiplier
    best = max(results, key=lambda r: r["f1"])
    best_mult = best["multiplier"]
    print(f"\n  Best mult={best_mult:.1f} → per-student thresholds:")
    base_thresh = 0.05
    for name in sorted(enrollment_embs):
        compactness = (spreads[name] - s_min) / (s_max - s_min + 1e-10)
        t = base_thresh * (1 - best_mult * compactness)
        tag = " *** MISSED" if name in best["missed"] else ""
        print(f"    {name:35s} spread={spreads[name]:.3f}  thresh={t:.4f}{tag}")

    return results


def strategy_cosine_fallback(enrollment_embs, val_detections, image_files, gt):
    """Strategy 3: If SVM says Unknown, check cosine sim to centroids as fallback."""
    print("\n" + "=" * 70)
    print("STRATEGY 3: COSINE SIMILARITY FALLBACK")
    print("=" * 70)

    model = FaceRecognitionModel(threshold=0.05)
    model.train(embeddings_dict=enrollment_embs)

    # Compute centroids
    centroids = {}
    for name, embs in enrollment_embs.items():
        c = np.mean(embs, axis=0)
        c = c / (np.linalg.norm(c) + 1e-10)
        centroids[name] = c
    centroid_names = sorted(centroids.keys())
    centroid_matrix = np.array([centroids[n] for n in centroid_names])  # (58, 512)

    cos_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    results = []

    for cos_thresh in cos_thresholds:
        predicted = set()
        fallback_recoveries = []

        for img_file in image_files:
            for face in val_detections[img_file]:
                emb = face["embedding"]
                label, conf = model.predict(emb)

                if label != "Unknown":
                    predicted.add(label)
                else:
                    # Fallback: cosine sim to all centroids
                    emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
                    sims = centroid_matrix @ emb_norm
                    best_idx = np.argmax(sims)
                    best_sim = sims[best_idx]
                    best_name = centroid_names[best_idx]

                    if best_sim >= cos_thresh:
                        predicted.add(best_name)
                        fallback_recoveries.append((best_name, float(best_sim), img_file))

        m = evaluate(predicted, gt)
        results.append({"cos_threshold": cos_thresh, "n_fallback_recoveries": len(fallback_recoveries), **m})
        print(f"  cos_thresh={cos_thresh:.2f}  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  "
              f"Prec={m['precision']:.1%}  Rec={m['recall']:.1%}  F1={m['f1']:.1%}  "
              f"fallbacks={len(fallback_recoveries)}  Missed={m['missed']}")

        if cos_thresh == 0.45:
            # Show what got recovered
            seen = set()
            for name, sim, img in sorted(fallback_recoveries, key=lambda x: -x[1]):
                if name not in seen:
                    print(f"      recovered: {name} (sim={sim:.3f} in {img})")
                    seen.add(name)

    return results


def strategy_combined(enrollment_embs, val_detections, image_files, gt):
    """Strategy 4: Lower threshold + cosine fallback combined."""
    print("\n" + "=" * 70)
    print("STRATEGY 4: COMBINED (lower threshold + cosine fallback)")
    print("=" * 70)

    # Compute centroids
    centroids = {}
    for name, embs in enrollment_embs.items():
        c = np.mean(embs, axis=0)
        c = c / (np.linalg.norm(c) + 1e-10)
        centroids[name] = c
    centroid_names = sorted(centroids.keys())
    centroid_matrix = np.array([centroids[n] for n in centroid_names])

    configs = [
        (0.03, 0.50),
        (0.03, 0.45),
        (0.035, 0.50),
        (0.035, 0.45),
        (0.04, 0.50),
        (0.04, 0.45),
    ]
    results = []

    for svm_thresh, cos_thresh in configs:
        model = FaceRecognitionModel(threshold=svm_thresh)
        model.train(embeddings_dict=enrollment_embs)

        predicted = set()
        for img_file in image_files:
            for face in val_detections[img_file]:
                emb = face["embedding"]
                label, conf = model.predict(emb)

                if label != "Unknown":
                    predicted.add(label)
                else:
                    emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
                    sims = centroid_matrix @ emb_norm
                    best_idx = np.argmax(sims)
                    best_sim = sims[best_idx]
                    best_name = centroid_names[best_idx]
                    if best_sim >= cos_thresh:
                        predicted.add(best_name)

        m = evaluate(predicted, gt)
        results.append({"svm_threshold": svm_thresh, "cos_threshold": cos_thresh, **m})
        print(f"  svm={svm_thresh:.3f} cos={cos_thresh:.2f}  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  "
              f"Prec={m['precision']:.1%}  Rec={m['recall']:.1%}  F1={m['f1']:.1%}  "
              f"Missed={m['missed']}")

    return results


def main():
    print("=" * 70)
    print("SVM TUNING BENCHMARK")
    print("=" * 70)

    enrollment_embs, val_detections, image_files = load_cache()
    gt = load_ground_truth()

    gt_present = {n for n, s in gt.items() if s == "P"}
    print(f"Ground truth: {len(gt_present)} present, {len(gt) - len(gt_present)} absent")

    r1 = strategy_threshold_sweep(enrollment_embs, val_detections, image_files, gt)
    r2 = strategy_adaptive_threshold(enrollment_embs, val_detections, image_files, gt)
    r3 = strategy_cosine_fallback(enrollment_embs, val_detections, image_files, gt)
    r4 = strategy_combined(enrollment_embs, val_detections, image_files, gt)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: BEST RESULT PER STRATEGY")
    print("=" * 70)
    for label, res in [("Threshold sweep", r1), ("Adaptive thresh", r2),
                       ("Cosine fallback", r3), ("Combined", r4)]:
        best = max(res, key=lambda r: (r["f1"], r["recall"]))
        print(f"  {label:20s}  TP={best['tp']}  FP={best['fp']}  FN={best['fn']}  "
              f"Prec={best['precision']:.1%}  Rec={best['recall']:.1%}  F1={best['f1']:.1%}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {"threshold_sweep": r1, "adaptive_threshold": r2,
           "cosine_fallback": r3, "combined": r4}
    with open(os.path.join(RESULTS_DIR, "bench_svm_tuning.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to results/bench_svm_tuning.json")


if __name__ == "__main__":
    main()
