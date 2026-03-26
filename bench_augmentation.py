"""
bench_augmentation.py - Data augmentation impact benchmark.

Tests the effect of augmenting training data on LOO accuracy and classroom recall.
For n_aug in [0, 2, 5, 10, 20]: augment each student's images, extract embeddings,
train SVM, and evaluate.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import gc
import json
import pickle
import argparse
import numpy as np
from PIL import Image, ImageOps
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from face_model import FaceRecognitionModel
from benchmark_detection import load_ground_truth, DATASET_DIR
from augment import generate_augmented_images
from cache_detections import load_cache

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")


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


def stratified_cv(embeddings_dict, n_folds=5):
    X, y = [], []
    for student, embs in embeddings_dict.items():
        for emb in embs:
            X.append(emb)
            y.append(student)
    X = np.array(X)
    y = np.array(y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    correct, total = 0, 0
    for train_idx, test_idx in skf.split(X, y_enc):
        svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
        svm.fit(X[train_idx], y_enc[train_idx])
        preds = svm.predict(X[test_idx])
        correct += (preds == y_enc[test_idx]).sum()
        total += len(test_idx)
    return correct / total if total > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Augmentation impact benchmark")
    parser.add_argument("--fast", action="store_true",
                        help="Use 5-fold CV instead of full LOO for n_aug > 0")
    args = parser.parse_args()

    print("=" * 70)
    print("DATA AUGMENTATION IMPACT BENCHMARK")
    if args.fast:
        print("  (--fast mode: 5-fold CV for n_aug > 0)")
    print("=" * 70)

    enrollment_embs, val_detections, image_files = load_cache()
    gt = load_ground_truth()

    # Build list of student dirs (images loaded lazily per-student to save memory)
    student_dirs = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )
    print(f"\n  {len(student_dirs)} student directories found")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "bench_augmentation.json")

    model = FaceRecognitionModel(threshold=0.05)
    n_aug_values = [0, 2, 5, 10, 20]
    results = []

    for n_aug in n_aug_values:
        print(f"\n{'='*60}")
        print(f"  n_aug = {n_aug}")
        print(f"{'='*60}")

        cache_path = os.path.join(RESULTS_DIR, f"aug_cache_n{n_aug}.pkl")

        if os.path.exists(cache_path):
            # Load cached embeddings from a previous (possibly OOM-killed) run
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            embeddings_dict = cached["embeddings_dict"]
            total_augmented_attempted = cached["attempted"]
            total_augmented_success = cached["success"]
            print(f"  Loaded from cache: {cache_path}")
        elif n_aug == 0:
            # Use cached enrollment embeddings directly
            embeddings_dict = enrollment_embs
            total_augmented_attempted = 0
            total_augmented_success = 0
            # Save cache for consistency
            with open(cache_path, "wb") as f:
                pickle.dump({"embeddings_dict": embeddings_dict,
                             "attempted": 0, "success": 0}, f)
        else:
            # Extract embeddings with lazy per-student image loading
            embeddings_dict = {}
            total_augmented_attempted = 0
            total_augmented_success = 0

            for name in student_dirs:
                embs = list(enrollment_embs.get(name, []))  # start with cached originals
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

                    aug_imgs = generate_augmented_images(pil_img, n_aug)
                    del pil_img
                    for aug_img in aug_imgs:
                        total_augmented_attempted += 1
                        aug_rgb = np.array(aug_img)
                        aug_emb = model.get_embedding(aug_rgb)
                        if aug_emb is not None:
                            embs.append(aug_emb)
                            total_augmented_success += 1
                        del aug_rgb, aug_emb
                    del aug_imgs
                gc.collect()

                if embs:
                    embeddings_dict[name] = embs
                print(f"    {name}: {len(embs)} embs", flush=True)

            # Save cache so re-runs skip this level
            with open(cache_path, "wb") as f:
                pickle.dump({"embeddings_dict": embeddings_dict,
                             "attempted": total_augmented_attempted,
                             "success": total_augmented_success}, f)
            print(f"  Saved cache: {cache_path}")

            # Release InsightFace singleton to free ONNX runtime memory
            import face_model as fm
            fm._face_app = None
            gc.collect()

        n_total_embs = sum(len(e) for e in embeddings_dict.values())
        aug_rate = total_augmented_success / total_augmented_attempted if total_augmented_attempted > 0 else 0
        print(f"  Total embeddings: {n_total_embs}")
        if n_aug > 0:
            print(f"  Aug success: {total_augmented_success}/{total_augmented_attempted} ({aug_rate:.1%})")

        use_fast = args.fast and n_aug > 0
        if use_fast:
            print(f"  Running 5-fold stratified CV...")
            cv_acc = stratified_cv(embeddings_dict, n_folds=5)
            cv_type = "5-fold"
        else:
            print(f"  Running leave-one-out CV...")
            cv_acc = loo_cv(embeddings_dict)
            cv_type = "LOO"
        print(f"  {cv_type} accuracy: {cv_acc:.1%}")

        print(f"  Running classroom eval...")
        cm = classroom_eval_cached(embeddings_dict, val_detections, image_files, gt)
        print(f"  Classroom: Recall={cm['recall']:.1%} Precision={cm['precision']:.1%} F1={cm['f1']:.1%}")

        results.append({
            "n_aug": n_aug, "n_embeddings": n_total_embs,
            "aug_success_rate": float(aug_rate),
            "cv_type": cv_type, "cv_accuracy": float(cv_acc),
            "recall": cm["recall"], "precision": cm["precision"], "f1": cm["f1"],
            "tp": cm["tp"], "fp": cm["fp"], "fn": cm["fn"],
        })

        # Save results incrementally (don't wait for all levels)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        # Free embeddings_dict if not needed downstream
        del embeddings_dict
        gc.collect()

    # Summary
    print(f"\n{'='*90}")
    print("AUGMENTATION IMPACT RESULTS")
    print(f"{'='*90}")
    print(f"{'n_aug':>5} {'#Embs':>7} {'AugRate':>8} {'CV Type':>8} {'CV Acc':>8} "
          f"{'Recall':>8} {'Prec':>8} {'F1':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['n_aug']:>5} {r['n_embeddings']:>7} {r['aug_success_rate']:>7.1%} "
              f"{r['cv_type']:>8} {r['cv_accuracy']:>7.1%} "
              f"{r['recall']:>7.1%} {r['precision']:>7.1%} {r['f1']:>7.1%}")
    print(f"{'='*90}")
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
