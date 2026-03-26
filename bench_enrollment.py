"""
bench_enrollment.py - Enrollment count degradation curve.

Tests how many enrollment images per student are needed for good accuracy.
For k in [1,2,3,4,5]: select first k embeddings, train SVM, run LOO + classroom eval.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from face_model import FaceRecognitionModel
from benchmark_detection import load_ground_truth, VAL_DIR
from cache_detections import load_cache

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

N_TRIALS = 5


def classroom_eval_cached(embeddings_dict, val_detections, image_files, ground_truth, threshold=0.05):
    """Classroom eval using cached detections + a freshly trained SVM."""
    rec = FaceRecognitionModel(threshold=threshold)
    rec.train(embeddings_dict=embeddings_dict)

    gt_present = {n for n, s in ground_truth.items() if s == "P"}
    gt_absent = {n for n, s in ground_truth.items() if s == "A"}
    predicted_present = set()

    for img_file in image_files:
        for face in val_detections[img_file]:
            label, conf = rec.predict(face["embedding"])
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
    """Leave-one-out CV. Returns accuracy."""
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
    print("ENROLLMENT COUNT DEGRADATION BENCHMARK")
    print("=" * 70)

    enrollment_embs, val_detections, image_files = load_cache()
    gt = load_ground_truth()

    n_students = len(enrollment_embs)
    max_k = min(len(embs) for embs in enrollment_embs.values())
    print(f"  {n_students} students, min embeddings per student: {max_k}")

    k_values = [1, 2, 3, 4, 5]
    results = []

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"  k = {k} enrollment images per student")
        print(f"{'='*60}")

        n_trials = 1 if k == max_k else N_TRIALS
        loo_accs = []
        classroom_metrics_list = []

        for trial in range(n_trials):
            subset = {}
            for student, embs in enrollment_embs.items():
                if len(embs) <= k:
                    subset[student] = embs
                else:
                    selected_idx = random.sample(range(len(embs)), k)
                    subset[student] = [embs[i] for i in selected_idx]

            # LOO (skip for k=1)
            if k >= 2:
                acc = loo_cv(subset)
                loo_accs.append(acc)
                print(f"  Trial {trial+1}: LOO accuracy = {acc:.1%}")

            cm = classroom_eval_cached(subset, val_detections, image_files, gt)
            classroom_metrics_list.append(cm)
            print(f"  Trial {trial+1}: Classroom Recall={cm['recall']:.1%} "
                  f"Precision={cm['precision']:.1%} F1={cm['f1']:.1%}")

        result = {"k": k, "n_trials": n_trials}
        if loo_accs:
            result["loo_mean"] = float(np.mean(loo_accs))
            result["loo_std"] = float(np.std(loo_accs))
        else:
            result["loo_mean"] = None
            result["loo_std"] = None

        recalls = [m["recall"] for m in classroom_metrics_list]
        precisions = [m["precision"] for m in classroom_metrics_list]
        f1s = [m["f1"] for m in classroom_metrics_list]
        result["recall_mean"] = float(np.mean(recalls))
        result["recall_std"] = float(np.std(recalls))
        result["precision_mean"] = float(np.mean(precisions))
        result["precision_std"] = float(np.std(precisions))
        result["f1_mean"] = float(np.mean(f1s))
        result["f1_std"] = float(np.std(f1s))
        results.append(result)

    # Summary
    print(f"\n{'='*90}")
    print("ENROLLMENT DEGRADATION RESULTS")
    print(f"{'='*90}")
    print(f"{'k':>3}  {'LOO Acc':>12}  {'Recall':>14}  {'Precision':>14}  {'F1':>14}")
    print("-" * 65)
    for r in results:
        loo_str = f"{r['loo_mean']:.1%}±{r['loo_std']:.1%}" if r['loo_mean'] is not None else "N/A (k=1)"
        rec_str = f"{r['recall_mean']:.1%}±{r['recall_std']:.1%}"
        prec_str = f"{r['precision_mean']:.1%}±{r['precision_std']:.1%}"
        f1_str = f"{r['f1_mean']:.1%}±{r['f1_std']:.1%}"
        print(f"{r['k']:>3}  {loo_str:>12}  {rec_str:>14}  {prec_str:>14}  {f1_str:>14}")
    print(f"{'='*90}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "bench_enrollment.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
