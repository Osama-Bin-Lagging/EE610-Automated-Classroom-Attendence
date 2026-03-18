"""
benchmark.py - Unified benchmarking of face recognition approaches
Tests facenet-pytorch, InsightFace, and DeepFace on the student dataset
using leave-one-out cross-validation with SVM and KNN classifiers.
"""

import os
import sys
import time
import warnings
import numpy as np
from PIL import Image, ImageOps
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

RAW_DATASET = "course_project_dataset"

# ─── Image loading (reused from preprocess.py) ───

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


def load_image_rgb(path):
    """Load image as RGB numpy array with EXIF orientation fix."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def get_student_images():
    """Load all student images. Returns {student_name: [rgb_array, ...]}."""
    data = {}
    student_dirs = sorted(
        d for d in os.listdir(RAW_DATASET)
        if os.path.isdir(os.path.join(RAW_DATASET, d))
    )
    for student_name in student_dirs:
        student_path = os.path.join(RAW_DATASET, student_name)
        images = []
        for f in sorted(os.listdir(student_path)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".heif")):
                try:
                    rgb = load_image_rgb(os.path.join(student_path, f))
                    images.append(rgb)
                except Exception as e:
                    print(f"  Warning: could not load {student_name}/{f}: {e}")
        if images:
            data[student_name] = images
    return data


# ─── Embedding extractors ───

def extract_facenet_pytorch(student_images):
    """Extract embeddings using facenet-pytorch (MTCNN + InceptionResnetV1)."""
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import torch

    # MTCNN must run on CPU (MPS doesn't support adaptive_avg_pool2d)
    # But the embedding model runs on MPS for speed
    mtcnn_device = torch.device("cpu")
    model_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=20, device=mtcnn_device, post_process=True)
    model = InceptionResnetV1(pretrained="vggface2").eval().to(model_device)

    embeddings = {}  # {student: [embedding, ...]}
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

    return embeddings, detected, total


def extract_insightface(student_images):
    """Extract embeddings using InsightFace (RetinaFace + ArcFace)."""
    import insightface
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    embeddings = {}
    total, detected = 0, 0

    for student, images in student_images.items():
        embs = []
        for rgb in images:
            total += 1
            faces = app.get(rgb)
            if faces:
                detected += 1
                # Take the largest face
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                embs.append(face.embedding)
        if embs:
            embeddings[student] = embs

    return embeddings, detected, total


def extract_deepface(student_images, model_name="ArcFace", detector="retinaface"):
    """Extract embeddings using DeepFace."""
    from deepface import DeepFace
    import tempfile
    import cv2

    embeddings = {}
    total, detected = 0, 0

    for student, images in student_images.items():
        embs = []
        for rgb in images:
            total += 1
            # DeepFace needs a file path or numpy array
            try:
                # Convert RGB to BGR for DeepFace (it expects BGR or file path)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                result = DeepFace.represent(
                    bgr,
                    model_name=model_name,
                    detector_backend=detector,
                    enforce_detection=True,
                    align=True,
                )
                if result:
                    detected += 1
                    embs.append(np.array(result[0]["embedding"]))
            except Exception:
                pass
        if embs:
            embeddings[student] = embs

    return embeddings, detected, total


# ─── Leave-one-out cross-validation ───

def leave_one_out_cv(embeddings):
    """
    Run leave-one-out CV with SVM and KNN classifiers.
    embeddings: {student_name: [embedding_array, ...]}
    Returns: (svm_accuracy, knn_accuracy)
    """
    # Build flat arrays
    X, y = [], []
    indices = []  # (student, image_idx) for each sample
    for student, embs in embeddings.items():
        for i, emb in enumerate(embs):
            X.append(emb)
            y.append(student)
            indices.append((student, i))

    X = np.array(X)
    y = np.array(y)

    if len(X) < 2:
        return 0.0, 0.0

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    svm_correct, knn_correct, total = 0, 0, 0

    for test_idx in range(len(X)):
        X_train = np.delete(X, test_idx, axis=0)
        y_train = np.delete(y_encoded, test_idx)
        X_test = X[test_idx:test_idx + 1]
        y_test = y_encoded[test_idx]

        # SVM
        svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
        svm.fit(X_train, y_train)
        svm_pred = svm.predict(X_test)[0]
        if svm_pred == y_test:
            svm_correct += 1

        # KNN
        k = min(3, len(X_train))
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)[0]
        if knn_pred == y_test:
            knn_correct += 1

        total += 1

    svm_acc = svm_correct / total if total > 0 else 0
    knn_acc = knn_correct / total if total > 0 else 0
    return svm_acc, knn_acc


# ─── Main benchmark ───

def run_benchmark():
    print("Loading student images...")
    student_images = get_student_images()
    total_images = sum(len(v) for v in student_images.values())
    print(f"Loaded {len(student_images)} students, {total_images} images\n")

    results = []

    # 1. facenet-pytorch
    print("=" * 60)
    print("[1/4] facenet-pytorch (MTCNN + InceptionResnetV1)")
    print("=" * 60)
    t0 = time.time()
    embs, detected, total = extract_facenet_pytorch(student_images)
    t1 = time.time()
    print(f"  Faces detected: {detected}/{total} ({t1-t0:.1f}s)")
    svm_acc, knn_acc = leave_one_out_cv(embs)
    t2 = time.time()
    print(f"  SVM accuracy: {svm_acc:.1%}")
    print(f"  KNN accuracy: {knn_acc:.1%}")
    print(f"  CV time: {t2-t1:.1f}s")
    results.append(("facenet-pytorch", "MTCNN", "FaceNet-512d", svm_acc, knn_acc, detected, total))

    # 2. InsightFace
    print("\n" + "=" * 60)
    print("[2/4] InsightFace (RetinaFace + ArcFace buffalo_l)")
    print("=" * 60)
    t0 = time.time()
    embs, detected, total = extract_insightface(student_images)
    t1 = time.time()
    print(f"  Faces detected: {detected}/{total} ({t1-t0:.1f}s)")
    svm_acc, knn_acc = leave_one_out_cv(embs)
    t2 = time.time()
    print(f"  SVM accuracy: {svm_acc:.1%}")
    print(f"  KNN accuracy: {knn_acc:.1%}")
    print(f"  CV time: {t2-t1:.1f}s")
    results.append(("InsightFace", "RetinaFace", "ArcFace-512d", svm_acc, knn_acc, detected, total))

    # 3. DeepFace with ArcFace
    print("\n" + "=" * 60)
    print("[3/4] DeepFace (RetinaFace + ArcFace)")
    print("=" * 60)
    t0 = time.time()
    embs, detected, total = extract_deepface(student_images, "ArcFace", "retinaface")
    t1 = time.time()
    print(f"  Faces detected: {detected}/{total} ({t1-t0:.1f}s)")
    svm_acc, knn_acc = leave_one_out_cv(embs)
    t2 = time.time()
    print(f"  SVM accuracy: {svm_acc:.1%}")
    print(f"  KNN accuracy: {knn_acc:.1%}")
    print(f"  CV time: {t2-t1:.1f}s")
    results.append(("DeepFace-ArcFace", "RetinaFace", "ArcFace", svm_acc, knn_acc, detected, total))

    # 4. DeepFace with FaceNet
    print("\n" + "=" * 60)
    print("[4/4] DeepFace (MTCNN + FaceNet)")
    print("=" * 60)
    t0 = time.time()
    embs, detected, total = extract_deepface(student_images, "Facenet512", "mtcnn")
    t1 = time.time()
    print(f"  Faces detected: {detected}/{total} ({t1-t0:.1f}s)")
    svm_acc, knn_acc = leave_one_out_cv(embs)
    t2 = time.time()
    print(f"  SVM accuracy: {svm_acc:.1%}")
    print(f"  KNN accuracy: {knn_acc:.1%}")
    print(f"  CV time: {t2-t1:.1f}s")
    results.append(("DeepFace-FaceNet", "MTCNN", "FaceNet-512d", svm_acc, knn_acc, detected, total))

    # ─── Summary table ───
    print("\n\n" + "=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)
    header = f"{'Approach':<20} {'Detector':<12} {'Embedding':<14} {'SVM Acc':>8} {'KNN Acc':>8} {'Detected':>10}"
    print(header)
    print("-" * 90)
    for name, det, emb, svm, knn, d, t in results:
        print(f"{name:<20} {det:<12} {emb:<14} {svm:>7.1%} {knn:>7.1%} {d:>4}/{t:<4}")
    print("-" * 90)
    print(f"{'LBPH (baseline)':<20} {'Haar':<12} {'LBP-hist':<14} {'58.7%':>8} {'N/A':>8} {'288/290':>10}")
    print("=" * 90)

    # Find winner
    best = max(results, key=lambda r: r[3])  # by SVM accuracy
    print(f"\nWinner: {best[0]} with SVM accuracy {best[3]:.1%}")

    return results


if __name__ == "__main__":
    run_benchmark()
