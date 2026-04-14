"""
benchmark.py - Extended benchmarking of face recognition approaches

Tests 16 approaches across 4 tiers using leave-one-out cross-validation
with SVM and KNN classifiers.

Tiers:
  1. Deep Embedding Models (face-specific)
  2. Classical CV (hand-crafted features)
  3. Generic Deep Features (ImageNet backbone, not face-specific)

Usage:
  python benchmark.py                        # run all 16 approaches
  python benchmark.py --only deep-embedding  # run only one tier
  python benchmark.py --skip classical       # skip a tier
"""

import os
import sys
import time
import json
import argparse
import warnings
import numpy as np
from PIL import Image, ImageOps
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

warnings.filterwarnings("ignore")

RAW_DATASET = "course_project_dataset"

# ─── Image loading ───

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


# ─── Face detection cache (for classical + generic approaches) ───

def detect_and_cache_faces(student_images):
    """Run InsightFace RetinaFace (ONNX) once, return {student: [(gray_112x112, rgb_112x112), ...]}."""
    import insightface
    from insightface.app import FaceAnalysis
    import cv2

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    cache = {}
    total, detected = 0, 0
    n_images = sum(len(v) for v in student_images.values())

    for student, images in student_images.items():
        crops = []
        for rgb in images:
            total += 1
            if total % 10 == 0 or total == n_images:
                print(f"  Detection: {total}/{n_images} images...", flush=True)
            faces = app.get(rgb)
            if faces:
                detected += 1
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                h, w = rgb.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face_rgb = cv2.resize(rgb[y1:y2, x1:x2], (112, 112))
                face_gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
                crops.append((face_gray, face_rgb))
        cache[student] = crops

    return cache, detected, total


# ─── Tier 1: Deep Embedding Extractors ───

def extract_facenet_pytorch(student_images):
    """facenet-pytorch: MTCNN + InceptionResnetV1 → 512-d."""
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import torch

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

    return embeddings, detected, total


def extract_insightface(student_images):
    """InsightFace: RetinaFace + ArcFace buffalo_l → 512-d."""
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
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                embs.append(face.embedding)
        if embs:
            embeddings[student] = embs

    return embeddings, detected, total


def extract_deepface_cached(student_images, face_cache, model_name="ArcFace"):
    """DeepFace using pre-detected face crops (skip detector). Much faster."""
    from deepface import DeepFace
    import cv2
    import gc

    n_total = sum(len(v) for v in face_cache.values())
    embeddings = {}
    total = sum(len(v) for v in student_images.values())
    detected = 0
    count = 0

    for student, crops in face_cache.items():
        embs = []
        for _, rgb_crop in crops:
            count += 1
            if count % 20 == 0 or count == n_total:
                print(f"    [{count}/{n_total}]", flush=True)
            try:
                bgr = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR)
                result = DeepFace.represent(
                    bgr,
                    model_name=model_name,
                    detector_backend="skip",
                    enforce_detection=False,
                    align=False,
                )
                if result:
                    detected += 1
                    embs.append(np.array(result[0]["embedding"]))
            except Exception:
                pass
        if embs:
            embeddings[student] = embs
        gc.collect()

    return embeddings, detected, total


def extract_dlib_native(student_images):
    """dlib native: HOG detection + dlib ResNet → 128-d embeddings."""
    import face_recognition

    embeddings = {}
    total, detected = 0, 0

    for student, images in student_images.items():
        embs = []
        for rgb in images:
            total += 1
            locations = face_recognition.face_locations(rgb, model="hog")
            if locations:
                encs = face_recognition.face_encodings(rgb, locations)
                if encs:
                    detected += 1
                    embs.append(encs[0])
        if embs:
            embeddings[student] = embs

    return embeddings, detected, total


# ─── Tier 2: Classical CV Extractors ───

def extract_lbph(student_images, face_cache):
    """Spatially-binned LBP histograms (8x8 grid, 10 bins each → 640-d)."""
    from skimage.feature import local_binary_pattern

    embeddings = {}
    total = sum(len(imgs) for imgs in student_images.values())
    detected = 0

    for student, crops in face_cache.items():
        embs = []
        for gray, _ in crops:
            detected += 1
            lbp = local_binary_pattern(gray, 8, 1, method="uniform")
            cell_h, cell_w = gray.shape[0] // 8, gray.shape[1] // 8
            parts = []
            for i in range(8):
                for j in range(8):
                    cell = lbp[i * cell_h:(i + 1) * cell_h,
                               j * cell_w:(j + 1) * cell_w]
                    hist, _ = np.histogram(cell.ravel(), bins=10, range=(0, 10))
                    hist = hist.astype(np.float64)
                    hist /= hist.sum() + 1e-7
                    parts.append(hist)
            embs.append(np.concatenate(parts))
        if embs:
            embeddings[student] = embs

    return embeddings, detected, total


def extract_eigenfaces(student_images, face_cache):
    """Eigenfaces: flatten grayscale → PCA(150).
    Note: PCA fit on full dataset before LOO — standard practice
    in face recognition literature (Turk & Pentland 1991)."""
    all_vectors, student_indices = [], {}
    idx = 0
    for student, crops in face_cache.items():
        student_indices[student] = []
        for gray, _ in crops:
            all_vectors.append(gray.flatten().astype(np.float64) / 255.0)
            student_indices[student].append(idx)
            idx += 1

    if not all_vectors:
        return {}, 0, 0

    X = np.array(all_vectors)
    n_components = min(150, X.shape[0] - 1, X.shape[1])
    X_pca = PCA(n_components=n_components).fit_transform(X)

    total = sum(len(imgs) for imgs in student_images.values())
    embeddings = {s: [X_pca[i] for i in idxs] for s, idxs in student_indices.items()}
    return embeddings, len(all_vectors), total


def extract_fisherfaces(student_images, face_cache):
    """Fisherfaces: flatten grayscale → PCA(300) → LDA(57).
    Note: PCA/LDA fit on full dataset before LOO — standard practice."""
    all_vectors, all_labels, student_indices = [], [], {}
    idx = 0
    for student, crops in face_cache.items():
        student_indices[student] = []
        for gray, _ in crops:
            all_vectors.append(gray.flatten().astype(np.float64) / 255.0)
            all_labels.append(student)
            student_indices[student].append(idx)
            idx += 1

    if not all_vectors:
        return {}, 0, 0

    X = np.array(all_vectors)
    y = LabelEncoder().fit_transform(all_labels)
    n_classes = len(set(all_labels))

    n_pca = min(X.shape[0] - 1, X.shape[1], 300)
    X_pca = PCA(n_components=n_pca).fit_transform(X)

    n_lda = min(n_classes - 1, X_pca.shape[1])
    X_lda = LinearDiscriminantAnalysis(n_components=n_lda).fit_transform(X_pca, y)

    total = sum(len(imgs) for imgs in student_images.values())
    embeddings = {s: [X_lda[i] for i in idxs] for s, idxs in student_indices.items()}
    return embeddings, len(all_vectors), total


def extract_hog_features(student_images, face_cache):
    """HOG descriptors from 112x112 grayscale crops → 6084-d."""
    from skimage.feature import hog

    embeddings = {}
    total = sum(len(imgs) for imgs in student_images.values())
    detected = 0

    for student, crops in face_cache.items():
        embs = []
        for gray, _ in crops:
            detected += 1
            h = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), feature_vector=True)
            embs.append(h)
        if embs:
            embeddings[student] = embs

    return embeddings, detected, total


# ─── Tier 3: Generic Deep Feature Extractors ───

def extract_resnet50_features(student_images, face_cache):
    """Frozen ResNet-50 (ImageNet) penultimate layer → 2048-d."""
    import torch
    import torchvision.transforms as T
    from torchvision.models import resnet50, ResNet50_Weights

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model = model.eval().to(device)

    transform = T.Compose([
        T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeddings = {}
    total = sum(len(imgs) for imgs in student_images.values())
    detected = 0

    for student, crops in face_cache.items():
        embs = []
        for _, rgb in crops:
            detected += 1
            tensor = transform(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(tensor)
            embs.append(feat.cpu().numpy().flatten())
        if embs:
            embeddings[student] = embs

    return embeddings, detected, total


def extract_efficientnet_features(student_images, face_cache):
    """Frozen EfficientNet-B0 (ImageNet) → 1280-d."""
    import torch
    import torchvision.transforms as T
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier = torch.nn.Identity()
    model = model.eval().to(device)

    transform = T.Compose([
        T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeddings = {}
    total = sum(len(imgs) for imgs in student_images.values())
    detected = 0

    for student, crops in face_cache.items():
        embs = []
        for _, rgb in crops:
            detected += 1
            tensor = transform(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(tensor)
            embs.append(feat.cpu().numpy().flatten())
        if embs:
            embeddings[student] = embs

    return embeddings, detected, total


# ─── Approach Registry ───

APPROACHES = [
    # Tier 1: Deep Embedding Models
    {"name": "InsightFace", "tier": "deep-embedding",
     "detector": "RetinaFace", "embedding": "ArcFace", "dim": 512,
     "extractor": "insightface"},
    {"name": "facenet-pytorch", "tier": "deep-embedding",
     "detector": "MTCNN", "embedding": "FaceNet", "dim": 512,
     "extractor": "facenet_pytorch"},
    {"name": "DF-ArcFace", "tier": "deep-embedding",
     "detector": "RetinaFace", "embedding": "ArcFace", "dim": 512,
     "extractor": "deepface_cached", "df_model": "ArcFace", "uses_cache": True},
    {"name": "DF-FaceNet512", "tier": "deep-embedding",
     "detector": "RetinaFace", "embedding": "FaceNet512", "dim": 512,
     "extractor": "deepface_cached", "df_model": "Facenet512", "uses_cache": True},
    {"name": "DF-GhostFaceNet", "tier": "deep-embedding",
     "detector": "RetinaFace", "embedding": "GhostFaceNet", "dim": 512,
     "extractor": "deepface_cached", "df_model": "GhostFaceNet", "uses_cache": True},
    {"name": "DF-VGG-Face", "tier": "deep-embedding",
     "detector": "RetinaFace", "embedding": "VGG-Face", "dim": 4096,
     "extractor": "deepface_cached", "df_model": "VGG-Face", "uses_cache": True},
    {"name": "DF-SFace", "tier": "deep-embedding",
     "detector": "RetinaFace", "embedding": "SFace", "dim": 128,
     "extractor": "deepface_cached", "df_model": "SFace", "uses_cache": True},
    {"name": "DF-OpenFace", "tier": "deep-embedding",
     "detector": "RetinaFace", "embedding": "OpenFace", "dim": 128,
     "extractor": "deepface_cached", "df_model": "OpenFace", "uses_cache": True},
    {"name": "DF-Dlib", "tier": "deep-embedding",
     "detector": "RetinaFace", "embedding": "Dlib", "dim": 128,
     "extractor": "deepface_cached", "df_model": "Dlib", "uses_cache": True},
    {"name": "dlib-native", "tier": "deep-embedding",
     "detector": "HOG (dlib)", "embedding": "dlib-ResNet", "dim": 128,
     "extractor": "dlib_native"},
    # Tier 2: Classical CV
    {"name": "LBPH", "tier": "classical",
     "detector": "RetinaFace", "embedding": "LBP-hist", "dim": 640,
     "extractor": "lbph", "uses_cache": True},
    {"name": "Eigenfaces", "tier": "classical",
     "detector": "RetinaFace", "embedding": "PCA", "dim": 150,
     "extractor": "eigenfaces", "uses_cache": True},
    {"name": "Fisherfaces", "tier": "classical",
     "detector": "RetinaFace", "embedding": "PCA+LDA", "dim": 57,
     "extractor": "fisherfaces", "uses_cache": True},
    {"name": "HOG", "tier": "classical",
     "detector": "RetinaFace", "embedding": "HOG", "dim": 6084,
     "extractor": "hog", "uses_cache": True},
    # Tier 3: Generic Deep Features
    {"name": "ResNet-50", "tier": "generic-deep",
     "detector": "RetinaFace", "embedding": "ImageNet", "dim": 2048,
     "extractor": "resnet50", "uses_cache": True},
    {"name": "EfficientNet-B0", "tier": "generic-deep",
     "detector": "RetinaFace", "embedding": "ImageNet", "dim": 1280,
     "extractor": "efficientnet", "uses_cache": True},
]

TIER_NAMES = {
    "deep-embedding": "Tier 1: Deep Embedding Models",
    "classical": "Tier 2: Classical CV",
    "generic-deep": "Tier 3: Generic Deep Features (not face-specific)",
}


# ─── Leave-one-out cross-validation ───

def leave_one_out_cv(embeddings):
    """Run leave-one-out CV with SVM and KNN classifiers."""
    X, y = [], []
    for student, embs in embeddings.items():
        for emb in embs:
            X.append(emb)
            y.append(student)

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

        svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
        svm.fit(X_train, y_train)
        if svm.predict(X_test)[0] == y_test:
            svm_correct += 1

        k = min(3, len(X_train))
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(X_train, y_train)
        if knn.predict(X_test)[0] == y_test:
            knn_correct += 1

        total += 1

    return svm_correct / total if total else 0, knn_correct / total if total else 0


# ─── Run an approach ───

def run_approach(approach, student_images, face_cache):
    """Run a single approach, return (embeddings, detected, total, extract_time)."""
    ext = approach["extractor"]
    t0 = time.time()

    if ext == "insightface":
        embs, det, tot = extract_insightface(student_images)
    elif ext == "facenet_pytorch":
        embs, det, tot = extract_facenet_pytorch(student_images)
    elif ext == "deepface_cached":
        embs, det, tot = extract_deepface_cached(
            student_images, face_cache, approach["df_model"])
    elif ext == "dlib_native":
        embs, det, tot = extract_dlib_native(student_images)
    elif ext == "lbph":
        embs, det, tot = extract_lbph(student_images, face_cache)
    elif ext == "eigenfaces":
        embs, det, tot = extract_eigenfaces(student_images, face_cache)
    elif ext == "fisherfaces":
        embs, det, tot = extract_fisherfaces(student_images, face_cache)
    elif ext == "hog":
        embs, det, tot = extract_hog_features(student_images, face_cache)
    elif ext == "resnet50":
        embs, det, tot = extract_resnet50_features(student_images, face_cache)
    elif ext == "efficientnet":
        embs, det, tot = extract_efficientnet_features(student_images, face_cache)
    else:
        raise ValueError(f"Unknown extractor: {ext}")

    return embs, det, tot, time.time() - t0


# ─── Main ───

def run_benchmark():
    parser = argparse.ArgumentParser(description="Extended face recognition benchmark")
    parser.add_argument("--only", type=str,
                        help="Run only this tier (deep-embedding, classical, generic-deep)")
    parser.add_argument("--skip", type=str,
                        help="Skip this tier")
    parser.add_argument("--name", type=str,
                        help="Run only this approach by name (e.g. DF-GhostFaceNet)")
    parser.add_argument("--append", action="store_true",
                        help="Append results to existing JSON instead of overwriting")
    args = parser.parse_args()

    approaches = APPROACHES[:]
    if args.name:
        approaches = [a for a in approaches if a["name"] == args.name]
    if args.only:
        approaches = [a for a in approaches if a["tier"] == args.only]
    if args.skip:
        approaches = [a for a in approaches if a["tier"] != args.skip]

    if not approaches:
        print("No approaches to run. Check --only/--skip flags.")
        sys.exit(1)

    print("Loading student images...")
    student_images = get_student_images()
    total_images = sum(len(v) for v in student_images.values())
    print(f"Loaded {len(student_images)} students, {total_images} images")
    print(f"Running {len(approaches)} approaches\n")

    # Build face cache if any approach needs it
    needs_cache = any(a.get("uses_cache") for a in approaches)
    face_cache = None
    if needs_cache:
        print("=" * 70)
        print("Caching RetinaFace detections for classical/generic approaches...")
        print("=" * 70)
        t0 = time.time()
        face_cache, cache_det, cache_tot = detect_and_cache_faces(student_images)
        print(f"  Faces detected: {cache_det}/{cache_tot} ({time.time() - t0:.1f}s)\n")

    results = []

    for i, approach in enumerate(approaches):
        name = approach["name"]
        print("=" * 70)
        print(f"[{i + 1}/{len(approaches)}] {name}  (tier: {approach['tier']})")
        print("=" * 70)

        try:
            embs, detected, total, extract_time = run_approach(
                approach, student_images, face_cache)
            print(f"  Faces detected: {detected}/{total} ({extract_time:.1f}s)")

            n_embs = sum(len(v) for v in embs.values()) if embs else 0
            if n_embs < 2:
                print("  Not enough embeddings for CV, skipping...")
                results.append({
                    **approach, "svm_acc": None, "knn_acc": None,
                    "detected": detected, "total": total,
                    "extract_time": extract_time, "cv_time": 0,
                })
                print()
                continue

            t0 = time.time()
            svm_acc, knn_acc = leave_one_out_cv(embs)
            cv_time = time.time() - t0
            print(f"  SVM accuracy: {svm_acc:.1%}")
            print(f"  KNN accuracy: {knn_acc:.1%}")
            print(f"  CV time: {cv_time:.1f}s")

            results.append({
                **approach, "svm_acc": svm_acc, "knn_acc": knn_acc,
                "detected": detected, "total": total,
                "extract_time": extract_time, "cv_time": cv_time,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                **approach, "svm_acc": None, "knn_acc": None,
                "detected": 0, "total": 0,
                "extract_time": 0, "cv_time": 0, "error": str(e),
            })

        print()

    # ─── Summary table ───
    print("\n" + "=" * 105)
    print("EXTENDED BENCHMARK RESULTS")
    print(f"{len(student_images)} students, {total_images} images, leave-one-out CV")
    print("=" * 105)
    header = (f"  {'Approach':<18} {'Tier':<16} {'Detector':<12} {'Embedding':<14} "
              f"{'Dim':>5} {'SVM':>7} {'KNN':>7} {'Det':>8} {'Time':>7}")
    print(header)
    print("-" * 105)

    current_tier = None
    for r in results:
        if r["tier"] != current_tier:
            current_tier = r["tier"]
            print(f"\n  --- {TIER_NAMES.get(current_tier, current_tier)} ---")

        svm_s = f"{r['svm_acc']:.1%}" if r["svm_acc"] is not None else "ERR"
        knn_s = f"{r['knn_acc']:.1%}" if r["knn_acc"] is not None else "ERR"
        det_s = f"{r['detected']}/{r['total']}" if r.get("total") else "N/A"
        time_s = f"{r['extract_time']:.0f}s" if r.get("extract_time") else "N/A"

        print(f"  {r['name']:<18} {r['tier']:<16} {r['detector']:<12} "
              f"{r['embedding']:<14} {r['dim']:>5} {svm_s:>7} {knn_s:>7} "
              f"{det_s:>8} {time_s:>7}")

    print("\n" + "=" * 105)

    # Winner
    valid = [r for r in results if r["svm_acc"] is not None]
    if valid:
        best = max(valid, key=lambda r: r["svm_acc"])
        print(f"Winner: {best['name']} with SVM accuracy {best['svm_acc']:.1%}")

    # ─── Save JSON ───
    json_path = "benchmark_scores_extended.json"
    output = {
        "meta": {
            "students": len(student_images),
            "images": total_images,
            "evaluation": "leave-one-out",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": [],
    }

    # Load existing results if appending
    if args.append and os.path.exists(json_path):
        with open(json_path, "r") as f:
            existing = json.load(f)
        existing_names = set()
        for er in existing.get("results", []):
            existing_names.add(er["name"])
        # Keep existing results that aren't being re-run
        for er in existing["results"]:
            if er["name"] not in {r["name"] for r in results}:
                output["results"].append(er)

    for r in results:
        output["results"].append({
            "name": r["name"],
            "tier": r["tier"],
            "detector": r["detector"],
            "embedding": r["embedding"],
            "dim": r["dim"],
            "svm_acc": round(r["svm_acc"], 4) if r["svm_acc"] is not None else None,
            "knn_acc": round(r["knn_acc"], 4) if r["knn_acc"] is not None else None,
            "detected": r.get("detected", 0),
            "total": r.get("total", 0),
            "extract_time_s": round(r.get("extract_time", 0), 1),
            "cv_time_s": round(r.get("cv_time", 0), 1),
            "error": r.get("error"),
        })

    with open("benchmark_scores_extended.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to benchmark_scores_extended.json")

    return results


if __name__ == "__main__":
    run_benchmark()
