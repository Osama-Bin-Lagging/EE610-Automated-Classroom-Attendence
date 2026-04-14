"""
bench_pipeline.py - Compare alternative recognition methods on val images.

Tests 4 variants against current pipeline:
  A: ArcFace + SVM (current, det_size=640)
  B: ArcFace + SVM + cosine fallback
  C: ArcFace + SVM (det_size=1280)
  D: AdaFace + SVM
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(PROJECT_DIR, "adaface_repo"))

from face_model import FaceRecognitionModel, load_image_rgb, _get_face_app
from benchmark_detection import load_ground_truth, VAL_DIR
from cache_detections import load_cache
from run import detect_image, _iou, _verify_face, DET_SCORE_MIN, NMS_IOU_THRESH

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
THRESHOLD = 0.035


def compute_metrics(predicted, gt):
    gt_present = {n for n, s in gt.items() if s == "P"}
    gt_absent = {n for n, s in gt.items() if s == "A"}
    tp = len(predicted & gt_present)
    fp = len(predicted & gt_absent)
    fn = len(gt_present - predicted)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"tp": tp, "fp": fp, "fn": fn, "prec": prec, "rec": rec, "f1": f1,
            "missed": sorted(gt_present - predicted),
            "false_pos": sorted(predicted & gt_absent)}


def get_val_images():
    return sorted(
        os.path.join(VAL_DIR, f) for f in os.listdir(VAL_DIR)
        if f.upper().endswith(".JPG")
    )


def detect_all(image_paths, model, det_size=None):
    """Detect faces in all images. Optionally override det_size."""
    app = _get_face_app()
    old_det_size = None
    if det_size:
        old_det_size = app.det_model.input_size
        app.prepare(ctx_id=0, det_size=det_size)

    detections = {}
    for path in image_paths:
        dets, _ = detect_image(model, path)
        detections[os.path.basename(path)] = dets

    if old_det_size:
        app.prepare(ctx_id=0, det_size=old_det_size)

    return detections


def predict_svm(model, detections):
    """Predict with SVM only."""
    present = set()
    for img_file, faces in detections.items():
        for face in faces:
            label, conf = model.predict(face["embedding"])
            if label != "Unknown":
                present.add(label)
    return present


def predict_cosine_fallback(model, detections, enrollment_embs, cos_thresh=0.45):
    """Predict with SVM + cosine fallback."""
    centroids = {}
    for name, embs in enrollment_embs.items():
        c = np.mean(embs, axis=0)
        c = c / (np.linalg.norm(c) + 1e-10)
        centroids[name] = c
    names = sorted(centroids.keys())
    centroid_matrix = np.array([centroids[n] for n in names])

    present = set()
    for img_file, faces in detections.items():
        for face in faces:
            label, conf = model.predict(face["embedding"])
            if label != "Unknown":
                present.add(label)
            else:
                emb = face["embedding"]
                emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
                sims = centroid_matrix @ emb_norm
                best_idx = np.argmax(sims)
                if sims[best_idx] >= cos_thresh:
                    present.add(names[best_idx])
    return present


def variant_a(image_paths, gt):
    """A: ArcFace + SVM (current pipeline, det_size=640)."""
    enrollment_embs, _, _ = load_cache()
    model = FaceRecognitionModel(threshold=THRESHOLD)
    model.train(embeddings_dict=enrollment_embs)
    detections = detect_all(image_paths, model)
    present = predict_svm(model, detections)
    return compute_metrics(present, gt)


def variant_b(image_paths, gt):
    """B: ArcFace + SVM + cosine fallback (threshold=0.45)."""
    enrollment_embs, _, _ = load_cache()
    model = FaceRecognitionModel(threshold=THRESHOLD)
    model.train(embeddings_dict=enrollment_embs)
    detections = detect_all(image_paths, model)
    present = predict_cosine_fallback(model, detections, enrollment_embs, cos_thresh=0.45)
    return compute_metrics(present, gt)


def variant_c(image_paths, gt):
    """C: ArcFace + SVM (det_size=1280)."""
    enrollment_embs, _, _ = load_cache()
    model = FaceRecognitionModel(threshold=THRESHOLD)
    model.train(embeddings_dict=enrollment_embs)
    detections = detect_all(image_paths, model, det_size=(1280, 1280))
    present = predict_svm(model, detections)
    return compute_metrics(present, gt)


def variant_d(image_paths, gt):
    """D: AdaFace + SVM."""
    import torch
    from insightface.utils import face_align

    import net

    ckpt = os.path.join(PROJECT_DIR, "adaface_repo", "pretrained", "adaface_ir50_ms1mv2.ckpt")
    ada_model = net.build_model("ir_50")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    ada_model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")})
    ada_model.eval()

    def ada_embed(rgb_112):
        bgr = ((rgb_112[:, :, ::-1] / 255.0) - 0.5) / 0.5
        t = torch.tensor([bgr.transpose(2, 0, 1)]).float()
        with torch.no_grad():
            feat, _ = ada_model(t)
        return feat[0].numpy()

    # Extract AdaFace enrollment embeddings
    print("    Extracting AdaFace enrollment embeddings...")
    app = _get_face_app()
    dataset_dir = os.path.join(PROJECT_DIR, "course_project_dataset")
    ada_enrollment = {}
    for student in sorted(d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))):
        embs = []
        student_path = os.path.join(dataset_dir, student)
        for fn in sorted(f for f in os.listdir(student_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".heif"))):
            try:
                rgb = load_image_rgb(os.path.join(student_path, fn))
                faces = app.get(rgb)
                if not faces:
                    continue
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                if not hasattr(face, "kps") or face.kps is None:
                    continue
                aligned = face_align.norm_crop(rgb, face.kps.astype(np.float32), image_size=112)
                embs.append(ada_embed(aligned))
            except Exception:
                continue
        if embs:
            ada_enrollment[student] = embs

    # Train SVM on AdaFace embeddings
    svm_model = FaceRecognitionModel(threshold=THRESHOLD)
    svm_model.train(embeddings_dict=ada_enrollment)

    # Detect + extract AdaFace embeddings for val images
    arcface_model = FaceRecognitionModel(threshold=THRESHOLD)
    arcface_model.load(os.path.join(PROJECT_DIR, "face_database_aug.pkl"))
    detections = detect_all(image_paths, arcface_model)

    # Re-embed each detection with AdaFace
    present = set()
    for img_file, faces in detections.items():
        rgb = load_image_rgb(os.path.join(VAL_DIR, img_file))
        for face in faces:
            kps = face.get("kps")
            if kps is None:
                continue
            aligned = face_align.norm_crop(rgb, np.array(kps, dtype=np.float32), image_size=112)
            emb = ada_embed(aligned)
            label, conf = svm_model.predict(emb)
            if label != "Unknown":
                present.add(label)

    return compute_metrics(present, gt)


def main():
    gt = load_ground_truth()
    image_paths = get_val_images()
    gt_present = sum(1 for s in gt.values() if s == "P")
    gt_absent = sum(1 for s in gt.values() if s == "A")
    print(f"Ground truth: {gt_present}P / {gt_absent}A, {len(image_paths)} images\n")

    variants = [
        ("A: ArcFace+SVM (baseline)", variant_a),
        ("B: ArcFace+SVM+cosine fallback", variant_b),
        ("C: ArcFace+SVM det_size=1280", variant_c),
        ("D: AdaFace+SVM", variant_d),
    ]

    results = []
    for name, fn in variants:
        print(f"Running {name}...")
        r = fn(image_paths, gt)
        results.append((name, r))
        print(f"  TP={r['tp']} FP={r['fp']} FN={r['fn']} "
              f"Prec={r['prec']:.1%} Rec={r['rec']:.1%} F1={r['f1']:.1%}")
        if r["missed"]:
            print(f"  Missed: {r['missed']}")
        if r["false_pos"]:
            print(f"  FP: {r['false_pos']}")

    print(f"\n{'='*80}")
    print(f"{'Variant':<40s} {'TP':>3} {'FP':>3} {'FN':>3} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 80)
    for name, r in results:
        print(f"{name:<40s} {r['tp']:>3} {r['fp']:>3} {r['fn']:>3} "
              f"{r['prec']:>6.1%} {r['rec']:>6.1%} {r['f1']:>6.1%}")
    print("=" * 80)

    with open(os.path.join(RESULTS_DIR, "bench_pipeline.json"), "w") as f:
        json.dump([{"variant": n, **{k: v for k, v in r.items()}} for n, r in results], f, indent=2)
    print(f"\nSaved to results/bench_pipeline.json")


if __name__ == "__main__":
    main()
