"""
bench_reid.py - Benchmark person re-identification methods.

Compares three approaches:
  1. SVM-only (baseline): each face classified independently
  2. Embedding-clustering: agglomerative clustering on face embeddings
  3. Full re-ID: embedding + spatial anchors + relative position + Hungarian

Measures attendance accuracy against ground truth.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
from face_model import FaceRecognitionModel
from cache_detections import load_cache
from benchmark_detection import load_ground_truth
from reid import PersonReIdentifier, EmbeddingOnlyReIdentifier

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")


def evaluate(predicted_present: set, gt: dict) -> dict:
    """Compute TP/FP/FN/precision/recall/F1 against ground truth."""
    gt_present = {n for n, s in gt.items() if s == "P"}
    gt_absent = {n for n, s in gt.items() if s == "A"}

    tp = len(predicted_present & gt_present)
    fp = len(predicted_present & gt_absent)
    fn = len(gt_present - predicted_present)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "recognized": sorted(predicted_present),
        "missed": sorted(gt_present - predicted_present),
        "false_positives": sorted(predicted_present & gt_absent),
    }


def bench_svm_only(val_detections, image_files, predict_fn, gt):
    """Baseline: classify each face independently with SVM."""
    present = set()
    for img_file in image_files:
        for det in val_detections[img_file]:
            label, conf = predict_fn(det["embedding"])
            if label != "Unknown":
                present.add(label)
    return evaluate(present, gt)


def bench_embedding_clustering(val_detections, image_files, predict_fn, gt,
                                threshold=0.5):
    """Embedding-only re-ID via agglomerative clustering."""
    reid = EmbeddingOnlyReIdentifier(threshold=threshold)
    reid.process_all(val_detections, predict_fn, image_files)
    attendance = reid.get_attendance()
    present = set(attendance.keys())
    result = evaluate(present, gt)
    result["n_clusters"] = len(reid.person_sets)
    result["threshold"] = threshold
    return result


def bench_full_reid(val_detections, image_files, predict_fn, gt):
    """Full re-ID with all methods."""
    reid = PersonReIdentifier()
    person_sets = reid.process_all(val_detections, predict_fn, image_files)
    attendance = reid.get_attendance()
    present = set(attendance.keys())
    result = evaluate(present, gt)
    stats = reid.get_results()["stats"]
    result["n_person_sets"] = stats["total_person_sets"]
    result["labeled_sets"] = stats["labeled"]
    result["unlabeled_sets"] = stats["unlabeled"]
    result["embedding_threshold"] = stats["embedding_threshold"]
    return result


def main():
    print("=" * 70)
    print("RE-ID BENCHMARK")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    enrollment_embs, val_detections, image_files = load_cache()
    recognizer = FaceRecognitionModel(threshold=0.05)
    recognizer.train(embeddings_dict=enrollment_embs)

    gt = load_ground_truth()
    gt_present = {n for n, s in gt.items() if s == "P"}
    print(f"Ground truth: {len(gt_present)} present students\n")

    all_results = {}

    # 1. SVM-only baseline
    print("-" * 70)
    print("1. SVM-ONLY BASELINE")
    print("-" * 70)
    r = bench_svm_only(val_detections, image_files, recognizer.predict, gt)
    all_results["svm_only"] = r
    print(f"   TP={r['tp']}  FP={r['fp']}  FN={r['fn']}")
    print(f"   Precision={r['precision']:.1%}  Recall={r['recall']:.1%}  F1={r['f1']:.1%}")
    print(f"   Missed: {r['missed']}")

    # 2. Embedding clustering (sweep thresholds)
    print("\n" + "-" * 70)
    print("2. EMBEDDING CLUSTERING (threshold sweep)")
    print("-" * 70)
    best_emb = None
    best_emb_f1 = -1
    emb_results = {}
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        r = bench_embedding_clustering(val_detections, image_files,
                                        recognizer.predict, gt, threshold=t)
        emb_results[str(t)] = r
        marker = ""
        if r["f1"] > best_emb_f1:
            best_emb_f1 = r["f1"]
            best_emb = r
            marker = " <-- best"
        print(f"   t={t:.1f}: TP={r['tp']} FP={r['fp']} FN={r['fn']} "
              f"F1={r['f1']:.1%} clusters={r['n_clusters']}{marker}")

    all_results["embedding_clustering"] = emb_results
    all_results["embedding_clustering_best"] = best_emb

    if best_emb:
        print(f"\n   Best embedding-only: t={best_emb['threshold']}")
        print(f"   Missed: {best_emb['missed']}")

    # 3. Full re-ID
    print("\n" + "-" * 70)
    print("3. FULL RE-ID (embedding + spatial + Hungarian)")
    print("-" * 70)
    r = bench_full_reid(val_detections, image_files, recognizer.predict, gt)
    all_results["full_reid"] = r
    print(f"   TP={r['tp']}  FP={r['fp']}  FN={r['fn']}")
    print(f"   Precision={r['precision']:.1%}  Recall={r['recall']:.1%}  F1={r['f1']:.1%}")
    print(f"   Person sets: {r['n_person_sets']} ({r['labeled_sets']} labeled, "
          f"{r['unlabeled_sets']} unknown)")
    print(f"   Embedding threshold (auto): {r['embedding_threshold']:.3f}")
    print(f"   Missed: {r['missed']}")

    # ── Comparison ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    svm_r = all_results["svm_only"]
    emb_r = all_results["embedding_clustering_best"]
    full_r = all_results["full_reid"]

    print(f"\n   {'Method':<30} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print(f"   {'-'*30} {'----':>4} {'----':>4} {'----':>4} {'-------':>7} {'-------':>7} {'-------':>7}")
    for name, r in [("SVM-only", svm_r), ("Embedding clustering", emb_r), ("Full Re-ID", full_r)]:
        print(f"   {name:<30} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
              f"{r['precision']:>6.1%} {r['recall']:>6.1%} {r['f1']:>6.1%}")

    gained_emb = set(emb_r["recognized"]) - set(svm_r["recognized"])
    gained_full = set(full_r["recognized"]) - set(svm_r["recognized"])
    if gained_emb:
        print(f"\n   Embedding clustering gained: {sorted(gained_emb)}")
    if gained_full:
        print(f"   Full re-ID gained: {sorted(gained_full)}")

    # Save results
    json_path = os.path.join(RESULTS_DIR, "bench_reid.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n   Results saved to {json_path}")


if __name__ == "__main__":
    main()
