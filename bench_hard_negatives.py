"""
bench_hard_negatives.py - Hard negative mining benchmark.

Phase 1: Pairwise cosine similarity matrix, top-10 most confusable pairs.
Phase 2: LOO with runner-up analysis (soft confusion even at 100% accuracy).
Phase 3: Targeted augmentation for top-10 pairs, retrain, measure margin improvement.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
from PIL import Image, ImageOps
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from face_model import FaceRecognitionModel
from benchmark_detection import load_ground_truth, DATASET_DIR
from augment import generate_augmented_images
from cache_detections import load_cache

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

N_TARGETED_AUG = 10


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def classroom_eval_cached(embeddings_dict, val_detections, image_files, ground_truth, threshold=0.05):
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


def main():
    print("=" * 70)
    print("HARD NEGATIVE MINING BENCHMARK")
    print("=" * 70)

    enrollment_embs, val_detections, image_files = load_cache()
    gt = load_ground_truth()
    students = sorted(enrollment_embs.keys())
    n_students = len(students)
    print(f"  {n_students} students")

    # Phase 1: Centroid similarity matrix
    print("\n" + "=" * 70)
    print("PHASE 1: PAIRWISE CENTROID SIMILARITY")
    print("=" * 70)

    centroids = {}
    for student, embs in enrollment_embs.items():
        c = np.mean(embs, axis=0)
        centroids[student] = c / (np.linalg.norm(c) + 1e-8)

    pairs = []
    for i in range(n_students):
        for j in range(i + 1, n_students):
            sim = cosine_similarity(centroids[students[i]], centroids[students[j]])
            pairs.append((students[i], students[j], sim))

    pairs.sort(key=lambda x: x[2], reverse=True)
    top_10 = pairs[:10]

    print(f"\nTop-10 most confusable pairs:")
    print(f"  {'Student A':<25} {'Student B':<25} {'Similarity':>10}")
    print("-" * 65)
    for s1, s2, sim in top_10:
        print(f"  {s1:<25} {s2:<25} {sim:>10.4f}")

    # Phase 2: LOO with runner-up
    print("\n" + "=" * 70)
    print("PHASE 2: LOO WITH RUNNER-UP ANALYSIS")
    print("=" * 70)

    X, y = [], []
    for student, embs in enrollment_embs.items():
        for emb in embs:
            X.append(emb)
            y.append(student)
    X = np.array(X)
    y = np.array(y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    correct, total = 0, 0
    runner_up_counts = {}
    margins = []

    for i in range(len(X)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y_enc, i)
        svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
        svm.fit(X_train, y_train)

        probs = svm.predict_proba(X[i:i+1])[0]
        sorted_idx = np.argsort(probs)[::-1]
        top1_prob = probs[sorted_idx[0]]
        top2_prob = probs[sorted_idx[1]]
        margins.append(float(top1_prob - top2_prob))

        true_label = le.inverse_transform([y_enc[i]])[0]
        runner_up = le.inverse_transform([sorted_idx[1]])[0]

        if sorted_idx[0] == y_enc[i]:
            correct += 1
        runner_up_counts[(true_label, runner_up)] = runner_up_counts.get((true_label, runner_up), 0) + 1
        total += 1

    loo_acc = correct / total
    print(f"\n  LOO Accuracy: {correct}/{total} = {loo_acc:.1%}")
    print(f"  Average margin: {np.mean(margins):.4f}, Min: {min(margins):.4f}")

    runner_up_sorted = sorted(runner_up_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top-10 runner-up confusions:")
    print(f"  {'True Student':<25} {'Runner-up':<25} {'Count':>6}")
    print("  " + "-" * 60)
    for (true, runner), count in runner_up_sorted[:10]:
        print(f"  {true:<25} {runner:<25} {count:>6}")

    # Phase 3: Targeted augmentation
    print("\n" + "=" * 70)
    print("PHASE 3: TARGETED AUGMENTATION FOR CONFUSABLE PAIRS")
    print("=" * 70)

    targeted_students = set()
    for s1, s2, _ in top_10:
        targeted_students.add(s1)
        targeted_students.add(s2)
    print(f"  Targeting {len(targeted_students)} students")

    model = FaceRecognitionModel(threshold=0.05)
    augmented_embs = {s: list(e) for s, e in enrollment_embs.items()}

    for student in targeted_students:
        student_path = os.path.join(DATASET_DIR, student)
        if not os.path.isdir(student_path):
            continue
        aug_count = 0
        for f in sorted(os.listdir(student_path)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".heif")):
                continue
            try:
                img = Image.open(os.path.join(student_path, f))
                img = ImageOps.exif_transpose(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                for aug_img in generate_augmented_images(img, N_TARGETED_AUG):
                    emb = model.get_embedding(np.array(aug_img))
                    if emb is not None:
                        augmented_embs[student].append(emb)
                        aug_count += 1
            except Exception:
                pass
        print(f"    {student}: +{aug_count} augmented")

    # Re-run LOO with augmented
    print("\n  Re-running LOO with targeted augmentation...")
    X_aug, y_aug = [], []
    for student, embs in augmented_embs.items():
        for emb in embs:
            X_aug.append(emb)
            y_aug.append(student)
    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)
    le_aug = LabelEncoder()
    y_aug_enc = le_aug.fit_transform(y_aug)

    correct_aug, total_aug = 0, 0
    margins_aug = []
    for i in range(len(X_aug)):
        X_train = np.delete(X_aug, i, axis=0)
        y_train = np.delete(y_aug_enc, i)
        svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
        svm.fit(X_train, y_train)
        probs = svm.predict_proba(X_aug[i:i+1])[0]
        sorted_idx = np.argsort(probs)[::-1]
        if sorted_idx[0] == y_aug_enc[i]:
            correct_aug += 1
        margins_aug.append(float(probs[sorted_idx[0]] - probs[sorted_idx[1]]))
        total_aug += 1

    loo_acc_aug = correct_aug / total_aug
    print(f"  LOO after aug: {correct_aug}/{total_aug} = {loo_acc_aug:.1%}")
    print(f"  Margin after aug: {np.mean(margins_aug):.4f}")

    # Classroom eval
    cm_before = classroom_eval_cached(enrollment_embs, val_detections, image_files, gt)
    cm_after = classroom_eval_cached(augmented_embs, val_detections, image_files, gt)
    print(f"\n  Classroom before: Recall={cm_before['recall']:.1%} F1={cm_before['f1']:.1%}")
    print(f"  Classroom after:  Recall={cm_after['recall']:.1%} F1={cm_after['f1']:.1%}")

    # Summary
    print(f"\n{'='*70}")
    print("HARD NEGATIVE MINING SUMMARY")
    print(f"{'='*70}")
    print(f"  LOO before: {loo_acc:.1%} (margin={np.mean(margins):.4f})")
    print(f"  LOO after:  {loo_acc_aug:.1%} (margin={np.mean(margins_aug):.4f})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_results = {
        "top_10_confusable_pairs": [
            {"student_a": s1, "student_b": s2, "cosine_sim": sim}
            for s1, s2, sim in top_10
        ],
        "loo_before": {"accuracy": loo_acc, "avg_margin": float(np.mean(margins)),
                       "min_margin": float(min(margins))},
        "loo_after": {"accuracy": loo_acc_aug, "avg_margin": float(np.mean(margins_aug)),
                      "min_margin": float(min(margins_aug))},
        "classroom_before": cm_before,
        "classroom_after": cm_after,
        "top_10_runner_up_confusions": [
            {"true": true, "runner_up": runner, "count": count}
            for (true, runner), count in runner_up_sorted[:10]
        ],
    }
    json_path = os.path.join(RESULTS_DIR, "bench_hard_negatives.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
