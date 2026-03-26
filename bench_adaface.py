"""
bench_adaface.py - Compare ArcFace vs AdaFace embeddings for attendance recognition.

AdaFace uses adaptive margin loss → quality-aware embeddings that handle
blurry/low-quality classroom crops better than standard ArcFace.

Pipeline:
  Detection: RetinaFace (cached, det_score >= 0.3)
  Alignment: insightface norm_crop using cached kps → 112x112
  Embedding: AdaFace IR50 (MS1MV2) → 512-d
  Classifier: SVM (same as baseline)

Compares: ArcFace-only vs AdaFace-only vs ensemble (avg SVM probs)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from insightface.utils import face_align

from face_model import FaceRecognitionModel, load_image_rgb, _get_face_app
from cache_detections import load_cache
from benchmark_detection import load_ground_truth, VAL_DIR, DATASET_DIR

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
ADAFACE_DIR = os.path.join(PROJECT_DIR, "adaface_repo")
ADAFACE_CKPT = os.path.join(ADAFACE_DIR, "pretrained", "adaface_ir50_ms1mv2.ckpt")


def load_adaface_model():
    """Load AdaFace IR50 model."""
    sys.path.insert(0, ADAFACE_DIR)
    import net
    model = net.build_model("ir_50")
    statedict = torch.load(ADAFACE_CKPT, map_location="cpu", weights_only=False)["state_dict"]
    model_statedict = {k[6:]: v for k, v in statedict.items() if k.startswith("model.")}
    model.load_state_dict(model_statedict)
    model.eval()
    return model


def rgb_to_adaface_tensor(rgb_112):
    """Convert 112x112 RGB numpy array to AdaFace input tensor."""
    bgr = ((rgb_112[:, :, ::-1] / 255.0) - 0.5) / 0.5
    tensor = torch.tensor([bgr.transpose(2, 0, 1)]).float()
    return tensor


def extract_adaface_embedding(model, rgb_112):
    """Extract 512-d AdaFace embedding from a 112x112 RGB crop."""
    tensor = rgb_to_adaface_tensor(rgb_112)
    with torch.no_grad():
        feat, _ = model(tensor)
    return feat[0].numpy()


def align_face_from_kps(rgb_image, kps):
    """Use insightface norm_crop to get 112x112 aligned face from landmarks."""
    kps_arr = np.array(kps, dtype=np.float32)
    aligned = face_align.norm_crop(rgb_image, kps_arr, image_size=112)
    # norm_crop returns same color space as input; if input is RGB, output is RGB
    return aligned


def extract_enrollment_adaface(adaface_model):
    """Extract AdaFace embeddings for all enrollment images."""
    app = _get_face_app()
    enrollment = {}

    student_dirs = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )

    for student_name in student_dirs:
        student_path = os.path.join(DATASET_DIR, student_name)
        embs = []
        image_files = sorted(
            f for f in os.listdir(student_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".heif"))
        )
        for img_file in image_files:
            try:
                rgb = load_image_rgb(os.path.join(student_path, img_file))
                faces = app.get(rgb)
                if not faces:
                    continue
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                if not hasattr(face, "kps") or face.kps is None:
                    continue
                aligned = align_face_from_kps(rgb, face.kps)
                emb = extract_adaface_embedding(adaface_model, aligned)
                embs.append(emb)
            except Exception as e:
                print(f"  ERROR: {student_name}/{img_file}: {e}")

        if embs:
            enrollment[student_name] = embs

    return enrollment


def evaluate(predicted_present, ground_truth):
    gt_present = {n for n, s in ground_truth.items() if s == "P"}
    gt_absent = {n for n, s in ground_truth.items() if s == "A"}
    tp = len(predicted_present & gt_present)
    fp = len(predicted_present & gt_absent)
    fn = len(gt_present - predicted_present)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    missed = sorted(gt_present - predicted_present)
    false_pos = sorted(predicted_present & gt_absent)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
            "missed": missed, "false_positives": false_pos}


def run_arcface_baseline(enrollment_embs, val_detections, image_files, gt, threshold=0.05):
    """Baseline: ArcFace embeddings (from cache) + SVM."""
    model = FaceRecognitionModel(threshold=threshold)
    model.train(embeddings_dict=enrollment_embs)

    predicted = set()
    for img_file in image_files:
        for face in val_detections[img_file]:
            label, conf = model.predict(face["embedding"])
            if label != "Unknown":
                predicted.add(label)

    return evaluate(predicted, gt), model


def extract_val_adaface_embeddings(adaface_model, val_detections, image_files):
    """Extract AdaFace embeddings for all cached val faces that have kps."""
    val_ada_embs = {}  # {img_file: [{"embedding": np.array, ...}, ...]}

    for img_file in image_files:
        img_path = os.path.join(VAL_DIR, img_file)
        rgb = load_image_rgb(img_path)
        ada_faces = []

        for face in val_detections[img_file]:
            kps = face.get("kps")
            if kps is None:
                continue
            aligned = align_face_from_kps(rgb, kps)
            emb = extract_adaface_embedding(adaface_model, aligned)
            ada_faces.append({
                "embedding": emb,
                "bbox": face["bbox"],
                "det_score": face["det_score"],
            })

        val_ada_embs[img_file] = ada_faces

    return val_ada_embs


def run_adaface_only(ada_enrollment, ada_val_embs, image_files, gt, threshold=0.05):
    """AdaFace embeddings + SVM."""
    X, y = [], []
    for student, embs in ada_enrollment.items():
        for emb in embs:
            X.append(emb)
            y.append(student)
    X = np.array(X)
    y = np.array(y)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    svm.fit(X, y_enc)

    predicted = set()
    for img_file in image_files:
        for face in ada_val_embs[img_file]:
            probs = svm.predict_proba(face["embedding"].reshape(1, -1))[0]
            max_idx = np.argmax(probs)
            conf = probs[max_idx]
            label = le.inverse_transform([max_idx])[0]
            if conf >= threshold:
                predicted.add(label)

    return evaluate(predicted, gt), svm, le


def run_ensemble(arc_model, ada_svm, ada_le, enrollment_embs, ada_enrollment,
                 val_detections, ada_val_embs, image_files, gt, threshold=0.05):
    """Ensemble: average SVM probabilities from ArcFace and AdaFace."""
    # Build mapping from AdaFace label encoder to ArcFace label encoder
    arc_labels = list(arc_model.label_encoder.classes_)
    ada_labels = list(ada_le.classes_)

    # Shared student set
    shared = sorted(set(arc_labels) & set(ada_labels))

    predicted = set()
    for img_file in image_files:
        arc_faces = val_detections[img_file]
        ada_faces = ada_val_embs[img_file]

        # Match faces by index (same detection, same order)
        n_faces = min(len(arc_faces), len(ada_faces))
        for i in range(n_faces):
            arc_emb = arc_faces[i]["embedding"]
            ada_emb = ada_faces[i]["embedding"]

            arc_probs = arc_model.svm.predict_proba(arc_emb.reshape(1, -1))[0]
            ada_probs = ada_svm.predict_proba(ada_emb.reshape(1, -1))[0]

            # Average probs over shared labels
            best_label = None
            best_avg = 0
            for name in shared:
                arc_idx = np.where(arc_model.label_encoder.classes_ == name)[0][0]
                ada_idx = np.where(ada_le.classes_ == name)[0][0]
                avg = (arc_probs[arc_idx] + ada_probs[ada_idx]) / 2
                if avg > best_avg:
                    best_avg = avg
                    best_label = name

            if best_label and best_avg >= threshold:
                predicted.add(best_label)

    return evaluate(predicted, gt)


def main():
    print("=" * 70)
    print("ADAFACE BENCHMARK")
    print("=" * 70)

    # Load cached detections (with kps and det_score filtering)
    enrollment_embs, val_detections, image_files = load_cache()
    gt = load_ground_truth()

    gt_present = {n for n, s in gt.items() if s == "P"}
    print(f"Ground truth: {len(gt_present)} present, {len(gt) - len(gt_present)} absent")

    # Check that cache has kps
    has_kps = sum(
        1 for img in image_files for f in val_detections[img] if f.get("kps") is not None
    )
    total = sum(len(val_detections[img]) for img in image_files)
    print(f"Val faces with kps: {has_kps}/{total}")

    if has_kps == 0:
        print("ERROR: No kps in cache. Regenerate: python cache_detections.py")
        return

    # 1. ArcFace baseline
    print("\n" + "-" * 70)
    print("1. ArcFace baseline (from cache)")
    print("-" * 70)
    arc_result, arc_model = run_arcface_baseline(enrollment_embs, val_detections, image_files, gt)
    print(f"  TP={arc_result['tp']}  FP={arc_result['fp']}  FN={arc_result['fn']}  "
          f"Prec={arc_result['precision']:.1%}  Rec={arc_result['recall']:.1%}  F1={arc_result['f1']:.1%}")
    print(f"  Missed: {arc_result['missed']}")

    # 2. Load AdaFace and extract embeddings
    print("\n" + "-" * 70)
    print("2. Loading AdaFace IR50 (MS1MV2)...")
    print("-" * 70)
    adaface_model = load_adaface_model()
    print("  Model loaded.")

    print("\n  Extracting AdaFace enrollment embeddings...")
    ada_enrollment = extract_enrollment_adaface(adaface_model)
    n_ada_enroll = sum(len(v) for v in ada_enrollment.values())
    print(f"  {len(ada_enrollment)} students, {n_ada_enroll} embeddings")

    print("\n  Extracting AdaFace val embeddings...")
    ada_val_embs = extract_val_adaface_embeddings(adaface_model, val_detections, image_files)
    n_ada_val = sum(len(v) for v in ada_val_embs.values())
    print(f"  {n_ada_val} val face embeddings")

    # 3. AdaFace-only
    print("\n" + "-" * 70)
    print("3. AdaFace-only + SVM")
    print("-" * 70)
    ada_result, ada_svm, ada_le = run_adaface_only(ada_enrollment, ada_val_embs, image_files, gt)
    print(f"  TP={ada_result['tp']}  FP={ada_result['fp']}  FN={ada_result['fn']}  "
          f"Prec={ada_result['precision']:.1%}  Rec={ada_result['recall']:.1%}  F1={ada_result['f1']:.1%}")
    print(f"  Missed: {ada_result['missed']}")

    # 4. Ensemble (avg SVM probs)
    print("\n" + "-" * 70)
    print("4. Ensemble (avg ArcFace + AdaFace SVM probs)")
    print("-" * 70)
    ens_result = run_ensemble(
        arc_model, ada_svm, ada_le,
        enrollment_embs, ada_enrollment,
        val_detections, ada_val_embs,
        image_files, gt,
    )
    print(f"  TP={ens_result['tp']}  FP={ens_result['fp']}  FN={ens_result['fn']}  "
          f"Prec={ens_result['precision']:.1%}  Rec={ens_result['recall']:.1%}  F1={ens_result['f1']:.1%}")
    print(f"  Missed: {ens_result['missed']}")

    # 5. Threshold sweep for AdaFace
    print("\n" + "-" * 70)
    print("5. AdaFace threshold sweep")
    print("-" * 70)
    sweep_results = []
    for thresh in [0.01, 0.02, 0.03, 0.04, 0.05]:
        r, _, _ = run_adaface_only(ada_enrollment, ada_val_embs, image_files, gt, threshold=thresh)
        sweep_results.append({"threshold": thresh, **r})
        print(f"  thresh={thresh:.2f}  TP={r['tp']}  FP={r['fp']}  FN={r['fn']}  "
              f"Prec={r['precision']:.1%}  Rec={r['recall']:.1%}  F1={r['f1']:.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for label, r in [("ArcFace", arc_result), ("AdaFace", ada_result), ("Ensemble", ens_result)]:
        print(f"  {label:12s}  TP={r['tp']}  FP={r['fp']}  FN={r['fn']}  "
              f"Prec={r['precision']:.1%}  Rec={r['recall']:.1%}  F1={r['f1']:.1%}  "
              f"Missed={r['missed']}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "arcface_baseline": arc_result,
        "adaface_only": ada_result,
        "ensemble": ens_result,
        "adaface_threshold_sweep": sweep_results,
    }
    with open(os.path.join(RESULTS_DIR, "bench_adaface.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to results/bench_adaface.json")


if __name__ == "__main__":
    main()
