"""
bench_failure_analysis.py - Failure analysis of missed students.

Analyzes which students are missed, whether their faces were detected but
misclassified or never detected, and saves annotated validation images.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import cv2
import numpy as np
from face_model import FaceRecognitionModel, load_image_rgb
from benchmark_detection import load_ground_truth, VAL_DIR
from cache_detections import load_cache

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FAILURE_DIR = os.path.join(RESULTS_DIR, "failure_analysis")


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    print("=" * 70)
    print("FAILURE ANALYSIS BENCHMARK")
    print("=" * 70)

    # Load cache and model
    enrollment_embs, val_detections, image_files = load_cache()
    recognizer = FaceRecognitionModel(threshold=0.05)
    recognizer.load(os.path.join(PROJECT_DIR, "face_database.pkl"))

    gt = load_ground_truth()
    gt_present = {n for n, s in gt.items() if s == "P"}
    gt_absent = {n for n, s in gt.items() if s == "A"}
    print(f"Ground truth: {len(gt_present)} present, {len(gt_absent)} absent")

    image_paths = [os.path.join(VAL_DIR, f) for f in image_files]

    # Classify all cached detections
    all_detections = []
    predicted_present = set()

    for img_idx, img_file in enumerate(image_files):
        for face_dict in val_detections[img_file]:
            emb = face_dict["embedding"]
            label, conf = recognizer.predict(emb)
            bbox = face_dict["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            all_detections.append({
                "img_idx": img_idx,
                "img_file": img_file,
                "bbox": bbox,
                "label": label,
                "confidence": float(conf),
                "det_score": face_dict["det_score"],
                "area": area,
                "embedding": emb,
            })
            if label != "Unknown":
                predicted_present.add(label)

    tp = len(predicted_present & gt_present)
    fp = len(predicted_present & gt_absent)
    fn_count = len(gt_present - predicted_present)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn_count) if (tp + fn_count) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    missed = sorted(gt_present - predicted_present)
    false_positives = sorted(predicted_present & gt_absent)

    print(f"\nClassroom Results:")
    print(f"  TP={tp}  FP={fp}  FN={fn_count}")
    print(f"  Precision={precision:.1%}  Recall={recall:.1%}  F1={f1:.1%}")
    print(f"  Missed students: {missed}")
    if false_positives:
        print(f"  False positives: {false_positives}")

    # Per-student detection matrix
    print("\n" + "-" * 70)
    print("PER-STUDENT DETECTION MATRIX")
    print("-" * 70)

    student_detections = {}
    for det in all_detections:
        if det["label"] != "Unknown":
            student_detections.setdefault(det["label"], []).append(det)

    print(f"\n{'Student':<30} {'#Imgs':>6} {'AvgConf':>8} {'AvgArea':>9} {'AvgDet':>8}")
    print("-" * 65)
    for student in sorted(gt_present):
        dets = student_detections.get(student, [])
        n_imgs = len(set(d["img_idx"] for d in dets))
        avg_conf = np.mean([d["confidence"] for d in dets]) if dets else 0
        avg_area = np.mean([d["area"] for d in dets]) if dets else 0
        avg_det = np.mean([d["det_score"] for d in dets]) if dets else 0
        marker = " *** MISSED" if student in missed else ""
        print(f"{student:<30} {n_imgs:>6} {avg_conf:>7.1%} {avg_area:>9.0f} {avg_det:>7.2f}{marker}")

    # Deep dive on missed students
    if missed:
        print("\n" + "=" * 70)
        print("DEEP DIVE: MISSED STUDENTS")
        print("=" * 70)

        emb_array = np.array(recognizer.embeddings)
        emb_labels = recognizer.embedding_labels

        for missed_student in missed:
            print(f"\n--- {missed_student} ---")

            student_embs = [emb_array[i] for i, l in enumerate(emb_labels) if l == missed_student]
            if not student_embs:
                print(f"  No enrollment embeddings found!")
                continue

            centroid = np.mean(student_embs, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            print(f"  Enrollment embeddings: {len(student_embs)}")

            # Check all detected faces from cache
            face_matches = []
            for det in all_detections:
                sim = cosine_similarity(centroid, det["embedding"])
                face_matches.append({
                    "img_file": det["img_file"],
                    "similarity": sim,
                    "predicted_label": det["label"],
                    "confidence": det["confidence"],
                    "area": det["area"],
                })

            face_matches.sort(key=lambda x: x["similarity"], reverse=True)

            print(f"\n  Top-10 nearest faces (by cosine sim to enrollment centroid):")
            print(f"  {'Img':<15} {'Sim':>6} {'Predicted':<25} {'Conf':>6} {'Area':>8}")
            for m in face_matches[:10]:
                marker = " <-- LIKELY MATCH" if m["similarity"] > 0.3 else ""
                print(f"  {m['img_file']:<15} {m['similarity']:>.3f} "
                      f"{m['predicted_label']:<25} {m['confidence']:>5.1%} "
                      f"{m['area']:>8.0f}{marker}")

            best_sim = face_matches[0]["similarity"] if face_matches else 0
            if best_sim > 0.3:
                best = face_matches[0]
                print(f"\n  DIAGNOSIS: Face likely DETECTED but MISCLASSIFIED as '{best['predicted_label']}' "
                      f"(sim={best_sim:.3f})")
            else:
                print(f"\n  DIAGNOSIS: Face likely NEVER DETECTED (max sim={best_sim:.3f})")

    # Save annotated images
    print("\n" + "=" * 70)
    print("SAVING ANNOTATED IMAGES")
    print("=" * 70)
    os.makedirs(FAILURE_DIR, exist_ok=True)

    for img_file in image_files:
        img_path = os.path.join(VAL_DIR, img_file)
        rgb = load_image_rgb(img_path)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        img_dets = [d for d in all_detections if d["img_file"] == img_file]
        for det in img_dets:
            bbox = [int(c) for c in det["bbox"]]
            x1, y1, x2, y2 = bbox
            label = det["label"]
            conf = det["confidence"]
            if label != "Unknown":
                color = (0, 200, 0)
                name = label.split("(")[0].strip() if "(" in label else label
                text = f"{name[:18]} ({conf:.1%})"
            else:
                color = (0, 0, 255)
                text = f"Unknown ({conf:.1%})"
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(bgr, text, (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        out_path = os.path.join(FAILURE_DIR, img_file)
        cv2.imwrite(out_path, bgr)
        print(f"  Saved {img_file} ({len(img_dets)} faces)")

    # Per-image summary
    print("\n" + "-" * 70)
    print("PER-IMAGE FACE COUNTS")
    print("-" * 70)
    for img_idx, img_file in enumerate(image_files):
        img_dets = [d for d in all_detections if d["img_idx"] == img_idx]
        recognized = [d for d in img_dets if d["label"] != "Unknown"]
        unknown = [d for d in img_dets if d["label"] == "Unknown"]
        unique_students = set(d["label"] for d in recognized)
        print(f"  {img_file:<15} total={len(img_dets):>3}  recognized={len(recognized):>3}  "
              f"unknown={len(unknown):>3}  unique_students={len(unique_students):>3}")

    # Save JSON
    metrics = {
        "tp": tp, "fp": fp, "fn": fn_count,
        "precision": precision, "recall": recall, "f1": f1,
        "predicted_present": sorted(predicted_present),
        "missed": missed,
        "false_positives": false_positives,
    }

    json_results = {
        "metrics": metrics,
        "per_image": [
            {
                "image": img_file,
                "total_faces": len([d for d in all_detections if d["img_file"] == img_file]),
                "recognized": len([d for d in all_detections if d["img_file"] == img_file and d["label"] != "Unknown"]),
                "unknown": len([d for d in all_detections if d["img_file"] == img_file and d["label"] == "Unknown"]),
            }
            for img_file in image_files
        ],
        "student_detection_counts": {
            student: {
                "images_detected_in": len(set(d["img_idx"] for d in student_detections.get(student, []))),
                "total_detections": len(student_detections.get(student, [])),
                "avg_confidence": float(np.mean([d["confidence"] for d in student_detections[student]])) if student in student_detections else 0,
                "is_missed": student in missed,
            }
            for student in sorted(gt_present)
        },
    }
    json_path = os.path.join(RESULTS_DIR, "bench_failure_analysis.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {json_path}")
    print(f"Annotated images saved to {FAILURE_DIR}/")


if __name__ == "__main__":
    main()
