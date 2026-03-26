"""
bench_self_training.py - Self-training / pseudo-labeling benchmark.

Runs the model on val images, collects high-confidence predictions as pseudo-labels,
adds them to training data, retrains SVM, and evaluates.

Uses leave-1-image-out CV to avoid data leakage.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from face_model import FaceRecognitionModel
from benchmark_detection import load_ground_truth
from cache_detections import load_cache

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")


def classroom_eval_cached(recognizer, val_detections, image_files, ground_truth):
    gt_present = {n for n, s in ground_truth.items() if s == "P"}
    gt_absent = {n for n, s in ground_truth.items() if s == "A"}
    predicted_present = set()
    for img_file in image_files:
        for face in val_detections[img_file]:
            label, conf = recognizer.predict(face["embedding"])
            if label != "Unknown":
                predicted_present.add(label)
    tp = len(predicted_present & gt_present)
    fp = len(predicted_present & gt_absent)
    fn = len(gt_present - predicted_present)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def loo_cv(embeddings_dict):
    X, y = [], []
    for student, embs in embeddings_dict.items():
        for emb in embs:
            X.append(emb)
            y.append(student)
    X = np.array(X)
    y = np.array(y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    correct, total = 0, 0
    for i in range(len(X)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y_enc, i)
        svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
        svm.fit(X_train, y_train)
        if svm.predict(X[i:i+1])[0] == y_enc[i]:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0


def main():
    print("=" * 70)
    print("SELF-TRAINING / PSEUDO-LABELING BENCHMARK")
    print("=" * 70)

    enrollment_embs, val_detections, image_files = load_cache()
    gt = load_ground_truth()
    gt_present = {n for n, s in gt.items() if s == "P"}
    gt_absent = {n for n, s in gt.items() if s == "A"}

    n_enrollment = sum(len(v) for v in enrollment_embs.values())
    print(f"  {len(enrollment_embs)} students, {n_enrollment} enrollment embeddings")

    # Train base model
    base_model = FaceRecognitionModel(threshold=0.05)
    base_model.train(embeddings_dict=enrollment_embs)

    # Baseline
    print("\nBaseline classroom eval...")
    baseline = classroom_eval_cached(base_model, val_detections, image_files, gt)
    print(f"  Recall={baseline['recall']:.1%} Prec={baseline['precision']:.1%} F1={baseline['f1']:.1%}")

    print("\nLOO on enrollment embeddings...")
    loo_enrollment = loo_cv(enrollment_embs)
    print(f"  LOO accuracy: {loo_enrollment:.1%}")

    # Collect all pseudo-labels from cached detections
    def collect_pseudo(threshold):
        pseudo = []
        for img_idx, img_file in enumerate(image_files):
            for face in val_detections[img_file]:
                label, conf = base_model.predict(face["embedding"])
                if label != "Unknown" and conf >= threshold:
                    pseudo.append((img_idx, label, float(conf), face["embedding"]))
        return pseudo

    thresholds = [0.08, 0.10, 0.15]
    results = []

    for thresh in thresholds:
        print(f"\n{'='*60}")
        print(f"  Pseudo-label threshold: {thresh}")
        print(f"{'='*60}")

        all_pseudo = collect_pseudo(thresh)
        n_pseudo = len(all_pseudo)
        unique_labels = set(p[1] for p in all_pseudo)
        print(f"  Pseudo-labels: {n_pseudo} ({len(unique_labels)} unique students)")

        # Leave-1-image-out CV
        print(f"  Running leave-1-image-out CV...")
        all_predicted = set()

        for held_out_idx in range(len(image_files)):
            train_pseudo = [p for p in all_pseudo if p[0] != held_out_idx]

            augmented = {s: list(e) for s, e in enrollment_embs.items()}
            for _, label, conf, emb in train_pseudo:
                augmented.setdefault(label, []).append(emb)

            rec = FaceRecognitionModel(threshold=0.05)
            rec.train(embeddings_dict=augmented)

            held_out_file = image_files[held_out_idx]
            for face in val_detections[held_out_file]:
                label, conf = rec.predict(face["embedding"])
                if label != "Unknown":
                    all_predicted.add(label)

        tp = len(all_predicted & gt_present)
        fp = len(all_predicted & gt_absent)
        fn = len(gt_present - all_predicted)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        l1io = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
        print(f"  L1IO: Recall={recall:.1%} Prec={precision:.1%} F1={f1:.1%}")

        # Optimistic (circular)
        print(f"  Optimistic eval (circular)...")
        opt_embs = {s: list(e) for s, e in enrollment_embs.items()}
        for _, label, conf, emb in all_pseudo:
            opt_embs.setdefault(label, []).append(emb)
        rec_opt = FaceRecognitionModel(threshold=0.05)
        rec_opt.train(embeddings_dict=opt_embs)
        optimistic = classroom_eval_cached(rec_opt, val_detections, image_files, gt)
        print(f"  Optimistic: Recall={optimistic['recall']:.1%} F1={optimistic['f1']:.1%}")

        results.append({
            "threshold": thresh, "n_pseudo_labels": n_pseudo,
            "n_unique_students": len(unique_labels),
            "leave_1_image_out": l1io, "optimistic_circular": optimistic,
        })

    # Summary
    print(f"\n{'='*90}")
    print("SELF-TRAINING RESULTS")
    print(f"{'='*90}")
    print(f"\n  Baseline: LOO={loo_enrollment:.1%}, Recall={baseline['recall']:.1%} F1={baseline['f1']:.1%}")
    print(f"\n  {'Thresh':>6} {'#Pseudo':>8} {'L1IO Recall':>12} {'L1IO F1':>10} {'Opt Recall':>12} {'Opt F1':>10}")
    print("  " + "-" * 65)
    for r in results:
        print(f"  {r['threshold']:>6.2f} {r['n_pseudo_labels']:>8} "
              f"{r['leave_1_image_out']['recall']:>11.1%} {r['leave_1_image_out']['f1']:>9.1%} "
              f"{r['optimistic_circular']['recall']:>11.1%} {r['optimistic_circular']['f1']:>9.1%}")
    print(f"{'='*90}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_results = {
        "baseline": {"loo_accuracy": loo_enrollment, "classroom": baseline},
        "thresholds": results,
    }
    json_path = os.path.join(RESULTS_DIR, "bench_self_training.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
