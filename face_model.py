"""
face_model.py - Deep face recognition model using InsightFace
Wraps RetinaFace detection + ArcFace embeddings with SVM classifier.
"""

import os
import pickle
import numpy as np
from PIL import Image, ImageOps
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Try to import HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# Lazy-load InsightFace to avoid slow import on every module load
_face_app = None


def _get_face_app():
    """Lazy-initialize the InsightFace app (downloads model on first use)."""
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def load_image_rgb(path):
    """Load image as RGB numpy array with EXIF orientation fix."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


class FaceRecognitionModel:
    """InsightFace-based face recognition with SVM classifier."""

    def __init__(self, threshold=0.05):
        # With 58 classes, even correct predictions get low absolute probability
        # (e.g. 13% for top-1 vs ~2% for top-2). Threshold of 0.05 works well.
        self.threshold = threshold
        self.svm = None
        self.label_encoder = None
        self.labels = []
        self.embeddings = None      # raw embeddings array (N x 512)
        self.embedding_labels = []  # per-embedding student labels

    def detect_faces(self, rgb_image):
        """
        Detect all faces in an RGB image.
        Returns list of dicts: [{bbox: [x1,y1,x2,y2], embedding: np.array, ...}]
        """
        app = _get_face_app()
        faces = app.get(rgb_image)
        return faces

    def get_embedding(self, rgb_image):
        """
        Detect the largest face and return its 512-d embedding.
        Returns embedding array or None if no face detected.
        """
        faces = self.detect_faces(rgb_image)
        if not faces:
            return None
        # Pick the largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.embedding

    def train(self, processed_dataset_path=None, embeddings_dict=None):
        """
        Train SVM classifier on face embeddings.

        Can be called with either:
        - processed_dataset_path: path to dataset with student folders of images
        - embeddings_dict: pre-computed {student_name: [embedding, ...]}
        """
        if embeddings_dict is None:
            embeddings_dict = self._extract_embeddings(processed_dataset_path)

        # Build flat arrays
        X, y = [], []
        for student, embs in embeddings_dict.items():
            for emb in embs:
                X.append(emb)
                y.append(student)

        X = np.array(X)
        y = np.array(y)

        # Store raw embeddings for visualization
        self.embeddings = X
        self.embedding_labels = list(y)

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        self.svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
        self.svm.fit(X, y_encoded)

        self.labels = list(self.label_encoder.classes_)
        print(f"Trained on {len(self.labels)} students, {len(X)} total embeddings")

    def _extract_embeddings(self, dataset_path):
        """Extract embeddings from raw student image folders."""
        import json

        raw_dataset = dataset_path or "course_project_dataset"
        embeddings = {}

        student_dirs = sorted(
            d for d in os.listdir(raw_dataset)
            if os.path.isdir(os.path.join(raw_dataset, d))
        )

        for student_name in student_dirs:
            student_path = os.path.join(raw_dataset, student_name)
            embs = []
            image_files = sorted(
                f for f in os.listdir(student_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".heif"))
            )
            for img_file in image_files:
                try:
                    rgb = load_image_rgb(os.path.join(student_path, img_file))
                    emb = self.get_embedding(rgb)
                    if emb is not None:
                        embs.append(emb)
                    else:
                        print(f"  NO FACE: {student_name}/{img_file}")
                except Exception as e:
                    print(f"  ERROR: {student_name}/{img_file}: {e}")

            if embs:
                embeddings[student_name] = embs
            status = "OK" if len(embs) >= 3 else "LOW" if len(embs) > 0 else "FAIL"
            print(f"[{status}] {student_name}: {len(embs)}/{len(image_files)} faces")

        return embeddings

    def predict(self, embedding):
        """
        Predict identity from a face embedding.
        Returns: (label, confidence) where confidence is probability.
        If confidence < threshold, label will be "Unknown".
        """
        if self.svm is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        probs = self.svm.predict_proba(embedding.reshape(1, -1))[0]
        max_idx = np.argmax(probs)
        confidence = probs[max_idx]
        label = self.label_encoder.inverse_transform([max_idx])[0]

        if confidence < self.threshold:
            return "Unknown", confidence

        return label, confidence

    def save(self, path):
        """Save trained model + raw embeddings to disk."""
        data = {
            "svm": self.svm,
            "label_encoder": self.label_encoder,
            "labels": self.labels,
            "threshold": self.threshold,
            "embeddings": self.embeddings,
            "embedding_labels": self.embedding_labels,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load trained model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.svm = data["svm"]
        self.label_encoder = data["label_encoder"]
        self.labels = data["labels"]
        self.threshold = data.get("threshold", 0.5)
        self.embeddings = data.get("embeddings")
        self.embedding_labels = data.get("embedding_labels", [])
        print(f"Model loaded: {len(self.labels)} students")

    def leave_one_out_test(self, dataset_path=None):
        """
        Leave-one-out cross-validation.
        Returns accuracy.
        """
        embeddings = self._extract_embeddings(dataset_path)

        X, y = [], []
        for student, embs in embeddings.items():
            for emb in embs:
                X.append(emb)
                y.append(student)

        X = np.array(X)
        y = np.array(y)

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        correct, total = 0, 0
        for test_idx in range(len(X)):
            X_train = np.delete(X, test_idx, axis=0)
            y_train = np.delete(y_encoded, test_idx)
            X_test = X[test_idx:test_idx + 1]
            y_test = y_encoded[test_idx]

            svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
            svm.fit(X_train, y_train)
            pred = svm.predict(X_test)[0]
            if pred == y_test:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        print(f"\nLeave-One-Out: {correct}/{total} = {accuracy:.1%}")
        return accuracy
