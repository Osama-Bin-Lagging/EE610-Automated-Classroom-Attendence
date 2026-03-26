"""
bench_ensemble.py - Ensemble InsightFace + FaceNet benchmark.

Extracts embeddings from both models, tests three ensemble strategies:
1. Concatenation (1024-d) with single SVM
2. Average probability vectors from separate SVMs
3. Max confidence: pick whichever SVM is more confident
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from face_model import FaceRecognitionModel, load_image_rgb, _get_face_app
from benchmark_detection import load_ground_truth, VAL_DIR, DATASET_DIR
from cache_detections import load_cache

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")


def extract_facenet_embeddings(student_images):
    """Extract FaceNet embeddings. student_images: {name: [rgb_array, ...]}"""
    from facenet_pytorch import MTCNN, InceptionResnetV1

    mtcnn_device = torch.device("cpu")
    model_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=20, device=mtcnn_device, post_process=True)
    model = InceptionResnetV1(pretrained="vggface2").eval().to(model_device)

    embeddings = {}
    total, detected = 0, 0
    for student, images in student_images.items():
        embs = []
        for rgb in images:
            total += 1
            face_tensor = mtcnn(Image.fromarray(rgb))
            if face_tensor is not None:
                detected += 1
                with torch.no_grad():
                    emb = model(face_tensor.unsqueeze(0).to(model_device))
                embs.append(emb.cpu().numpy().flatten())
        if embs:
            embeddings[student] = embs
    print(f"  FaceNet: {detected}/{total} faces detected")
    return embeddings


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


def loo_cv_dual(insight_dict, facenet_dict, mode="avg"):
    """LOO CV with dual-model ensemble (avg probs or max conf)."""
    common = sorted(set(insight_dict.keys()) & set(facenet_dict.keys()))
    X_in, X_fn, y = [], [], []
    for student in common:
        n = min(len(insight_dict[student]), len(facenet_dict[student]))
        for i in range(n):
            X_in.append(insight_dict[student][i])
            X_fn.append(facenet_dict[student][i])
            y.append(student)

    X_in, X_fn = np.array(X_in), np.array(X_fn)
    y = np.array(y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    correct, total = 0, 0
    for i in range(len(X_in)):
        Xi_tr = np.delete(X_in, i, axis=0)
        Xf_tr = np.delete(X_fn, i, axis=0)
        y_tr = np.delete(y_enc, i)

        svm_in = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
        svm_in.fit(Xi_tr, y_tr)
        probs_in = svm_in.predict_proba(X_in[i:i+1])[0]

        svm_fn = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
        svm_fn.fit(Xf_tr, y_tr)
        probs_fn = svm_fn.predict_proba(X_fn[i:i+1])[0]

        if mode == "avg":
            pred = np.argmax((probs_in + probs_fn) / 2)
        else:  # max_conf
            pred = np.argmax(probs_in) if np.max(probs_in) >= np.max(probs_fn) else np.argmax(probs_fn)

        if pred == y_enc[i]:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0


def classroom_eval_ensemble(rec_insight, rec_facenet, strategy,
                            val_detections, image_files, ground_truth):
    """Classroom eval using cached detections. For FaceNet, uses crop from bbox."""
    from facenet_pytorch import MTCNN, InceptionResnetV1

    mtcnn_device = torch.device("cpu")
    model_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=20, device=mtcnn_device, post_process=True)
    facenet_model = InceptionResnetV1(pretrained="vggface2").eval().to(model_device)

    gt_present = {n for n, s in ground_truth.items() if s == "P"}
    gt_absent = {n for n, s in ground_truth.items() if s == "A"}
    predicted_present = set()

    for img_file in image_files:
        img_path = os.path.join(VAL_DIR, img_file)
        rgb = load_image_rgb(img_path)

        for face_dict in val_detections[img_file]:
            insight_emb = face_dict["embedding"]
            bbox = face_dict["bbox"]

            # Get FaceNet embedding from crop
            h, w = rgb.shape[:2]
            x1, y1, x2, y2 = [max(0, int(c)) for c in bbox]
            x2, y2 = min(w, x2), min(h, y2)
            bw, bh = x2 - x1, y2 - y1
            pad = int(max(bw, bh) * 0.3)
            cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
            cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
            crop = rgb[cy1:cy2, cx1:cx2]

            facenet_emb = None
            if crop.size > 0:
                face_tensor = mtcnn(Image.fromarray(crop))
                if face_tensor is not None:
                    with torch.no_grad():
                        facenet_emb = facenet_model(
                            face_tensor.unsqueeze(0).to(model_device)
                        ).cpu().numpy().flatten()

            if strategy == "insight_only":
                label, conf = rec_insight.predict(insight_emb)
            elif strategy == "facenet_only":
                if facenet_emb is None:
                    continue
                label, conf = rec_facenet.predict(facenet_emb)
            elif strategy == "avg_probs":
                probs_in = rec_insight.svm.predict_proba(insight_emb.reshape(1, -1))[0]
                if facenet_emb is not None:
                    probs_fn = rec_facenet.svm.predict_proba(facenet_emb.reshape(1, -1))[0]
                    avg = (probs_in + probs_fn) / 2
                else:
                    avg = probs_in
                max_idx = np.argmax(avg)
                conf = avg[max_idx]
                label = rec_insight.label_encoder.inverse_transform([max_idx])[0]
                if conf < rec_insight.threshold:
                    label = "Unknown"
            elif strategy == "max_conf":
                l_in, c_in = rec_insight.predict(insight_emb)
                if facenet_emb is not None:
                    l_fn, c_fn = rec_facenet.predict(facenet_emb)
                    label, conf = (l_fn, c_fn) if c_fn > c_in else (l_in, c_in)
                else:
                    label, conf = l_in, c_in
            else:
                continue

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
    print("ENSEMBLE BENCHMARK: InsightFace + FaceNet")
    print("=" * 70)

    insight_embs, val_detections, image_files = load_cache()
    gt = load_ground_truth()

    # Load student images for FaceNet extraction
    print("Loading student images for FaceNet...")
    student_dirs = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )
    student_images = {}
    for name in student_dirs:
        path = os.path.join(DATASET_DIR, name)
        imgs = []
        for f in sorted(os.listdir(path)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".heif")):
                try:
                    imgs.append(load_image_rgb(os.path.join(path, f)))
                except Exception:
                    pass
        if imgs:
            student_images[name] = imgs

    print("\nExtracting FaceNet embeddings...")
    facenet_embs = extract_facenet_embeddings(student_images)

    common = set(insight_embs.keys()) & set(facenet_embs.keys())
    print(f"  Common students: {len(common)}")

    results = {}

    print("\nLOO: InsightFace only...")
    results["insight_loo"] = loo_cv(insight_embs)
    print(f"  {results['insight_loo']:.1%}")

    print("LOO: FaceNet only...")
    results["facenet_loo"] = loo_cv(facenet_embs)
    print(f"  {results['facenet_loo']:.1%}")

    print("LOO: Concat (1024-d)...")
    concat_dict = {}
    for s in sorted(common):
        n = min(len(insight_embs[s]), len(facenet_embs[s]))
        concat_dict[s] = [np.concatenate([insight_embs[s][i], facenet_embs[s][i]]) for i in range(n)]
    results["concat_loo"] = loo_cv(concat_dict)
    print(f"  {results['concat_loo']:.1%}")

    print("LOO: Avg probabilities...")
    results["avg_probs_loo"] = loo_cv_dual(insight_embs, facenet_embs, "avg")
    print(f"  {results['avg_probs_loo']:.1%}")

    print("LOO: Max confidence...")
    results["max_conf_loo"] = loo_cv_dual(insight_embs, facenet_embs, "max")
    print(f"  {results['max_conf_loo']:.1%}")

    # Classroom eval
    rec_insight = FaceRecognitionModel(threshold=0.05)
    rec_insight.train(embeddings_dict=insight_embs)
    rec_facenet = FaceRecognitionModel(threshold=0.05)
    rec_facenet.train(embeddings_dict=facenet_embs)

    print("\nClassroom evaluation:")
    classroom_results = {}
    for strategy in ["insight_only", "facenet_only", "avg_probs", "max_conf"]:
        print(f"  {strategy}...", end=" ")
        m = classroom_eval_ensemble(rec_insight, rec_facenet, strategy,
                                    val_detections, image_files, gt)
        classroom_results[strategy] = m
        print(f"Recall={m['recall']:.1%} Prec={m['precision']:.1%} F1={m['f1']:.1%}")

    # Summary
    print(f"\n{'='*80}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*80}")
    print(f"\n  LOO Accuracy:")
    for k in ["insight_loo", "facenet_loo", "concat_loo", "avg_probs_loo", "max_conf_loo"]:
        print(f"    {k:<20} {results[k]:.1%}")
    print(f"\n  {'Strategy':<20} {'TP':>4} {'FP':>4} {'FN':>4} {'Recall':>8} {'Prec':>8} {'F1':>8}")
    print("  " + "-" * 55)
    for name, m in classroom_results.items():
        print(f"  {name:<20} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} "
              f"{m['recall']:>7.1%} {m['precision']:>7.1%} {m['f1']:>7.1%}")
    print(f"{'='*80}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_results = {
        "loo": {
            "insightface": results["insight_loo"], "facenet": results["facenet_loo"],
            "concat_1024d": results["concat_loo"], "avg_probs": results["avg_probs_loo"],
            "max_conf": results["max_conf_loo"],
        },
        "classroom": classroom_results,
    }
    json_path = os.path.join(RESULTS_DIR, "bench_ensemble.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
