"""
generate_attendance.py - Production attendance output using cross-image re-ID.

Loads cached detections, runs person re-identification across all classroom
images, and generates:
  - outputs/attendance.csv — Name, Status, Images_Detected_In, Max_Confidence
  - outputs/annotated/*.jpg — bbox overlays (green=recognized, red=unknown, gray=low det)
  - outputs/unknowns/ — crops of unknown person faces grouped by cluster
  - outputs/reid_results.json — full per-face results
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import cv2
import numpy as np
import pandas as pd
from face_model import FaceRecognitionModel, load_image_rgb
from cache_detections import load_cache
from benchmark_detection import load_ground_truth, VAL_DIR, DATASET_DIR
from reid import PersonReIdentifier

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")


def main():
    print("=" * 70)
    print("GENERATING ATTENDANCE WITH CROSS-IMAGE RE-ID")
    print("=" * 70)

    # Load cache and model
    enrollment_embs, val_detections, image_files = load_cache()
    recognizer = FaceRecognitionModel(threshold=0.05)
    recognizer.train(embeddings_dict=enrollment_embs)

    gt = load_ground_truth()
    gt_present = {n for n, s in gt.items() if s == "P"}
    gt_absent = {n for n, s in gt.items() if s == "A"}

    # Build class list from enrollment
    class_list = sorted(enrollment_embs.keys())

    print(f"\nEnrollment: {len(class_list)} students")
    print(f"Validation: {len(image_files)} images")
    print(f"Ground truth: {len(gt_present)} present, {len(gt_absent)} absent")

    # ── SVM-only baseline ─────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("SVM-ONLY BASELINE")
    print("-" * 70)

    svm_present = set()
    for img_file in image_files:
        for det in val_detections[img_file]:
            label, conf = recognizer.predict(det["embedding"])
            if label != "Unknown":
                svm_present.add(label)

    svm_tp = len(svm_present & gt_present)
    svm_fp = len(svm_present & gt_absent)
    svm_fn = len(gt_present - svm_present)
    print(f"SVM-only: TP={svm_tp} FP={svm_fp} FN={svm_fn}")
    print(f"  Recognized {len(svm_present)} unique students")

    # ── Full Re-ID ────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("FULL RE-ID PIPELINE")
    print("-" * 70)

    reid = PersonReIdentifier()
    person_sets = reid.process_all(val_detections, recognizer.predict, image_files)
    attendance = reid.get_attendance()
    results = reid.get_results()

    reid_present = set(attendance.keys())
    reid_tp = len(reid_present & gt_present)
    reid_fp = len(reid_present & gt_absent)
    reid_fn = len(gt_present - reid_present)
    reid_prec = reid_tp / (reid_tp + reid_fp) if (reid_tp + reid_fp) > 0 else 0
    reid_rec = reid_tp / (reid_tp + reid_fn) if (reid_tp + reid_fn) > 0 else 0
    reid_f1 = 2 * reid_prec * reid_rec / (reid_prec + reid_rec) if (reid_prec + reid_rec) > 0 else 0

    print(f"Re-ID: TP={reid_tp} FP={reid_fp} FN={reid_fn}")
    print(f"  Precision={reid_prec:.1%}  Recall={reid_rec:.1%}  F1={reid_f1:.1%}")
    print(f"  {results['stats']['total_person_sets']} person sets "
          f"({results['stats']['labeled']} labeled, "
          f"{results['stats']['unlabeled']} unknown)")
    print(f"  Embedding threshold: {results['stats']['embedding_threshold']:.3f}")

    gained = reid_present - svm_present
    lost = svm_present - reid_present
    if gained:
        print(f"\n  GAINED via re-ID: {sorted(gained)}")
    if lost:
        print(f"  LOST vs SVM-only: {sorted(lost)}")

    missed = sorted(gt_present - reid_present)
    if missed:
        print(f"\n  Still missed: {missed}")

    # ── Generate outputs ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING OUTPUTS")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "annotated"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "unknowns"), exist_ok=True)

    # 1. Attendance CSV
    rows = []
    for student in class_list:
        if student in attendance:
            att = attendance[student]
            rows.append({
                "Name": student,
                "Status": "Present",
                "Images_Detected_In": att["images_detected_in"],
                "Max_Confidence": f"{att['max_confidence']:.4f}",
            })
        else:
            rows.append({
                "Name": student,
                "Status": "Absent",
                "Images_Detected_In": 0,
                "Max_Confidence": "",
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "attendance.csv")
    df.to_csv(csv_path, index=False)
    present_count = sum(1 for r in rows if r["Status"] == "Present")
    absent_count = sum(1 for r in rows if r["Status"] == "Absent")
    print(f"\n  attendance.csv: {present_count} present, {absent_count} absent")

    # Build face-to-person lookup for annotation
    face_lookup = {}  # (img_file, face_idx) -> (label, person_id)
    for ps in person_sets:
        for face in ps.faces:
            face_lookup[(face.image_file, face.face_idx)] = (ps.label, ps.person_id)

    # 2. Annotated images
    for img_file in image_files:
        img_path = os.path.join(VAL_DIR, img_file)
        rgb = load_image_rgb(img_path)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        for fi, det in enumerate(val_detections[img_file]):
            bbox = [int(c) for c in det["bbox"]]
            x1, y1, x2, y2 = bbox
            det_score = det["det_score"]

            label_info = face_lookup.get((img_file, fi))
            if label_info:
                label, pid = label_info
            else:
                label, pid = "???", -1

            if label.startswith("Unknown Person"):
                color = (0, 0, 255)  # Red for unknown person
                text = f"#{pid} ({det_score:.2f})"
            elif label == "???":
                color = (128, 128, 128)  # Gray for unmatched
                text = f"? ({det_score:.2f})"
            else:
                color = (0, 200, 0)  # Green for recognized
                name = label.split("(")[0].strip() if "(" in label else label
                if len(name) > 18:
                    name = name[:16] + ".."
                text = f"{name}"

            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
            label_y = y1 - 8 if y1 - 8 > 15 else y2 + 18
            cv2.putText(bgr, text, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        out_path = os.path.join(OUTPUT_DIR, "annotated", f"annotated_{img_file}")
        cv2.imwrite(out_path, bgr)

    print(f"  annotated/: {len(image_files)} images saved")

    # 3. Unknown person crops (grouped by PersonSet)
    unknown_sets = [ps for ps in person_sets if ps.label.startswith("Unknown Person")]
    crop_count = 0
    for ps in unknown_sets:
        set_dir = os.path.join(OUTPUT_DIR, "unknowns", f"person_{ps.person_id}")
        os.makedirs(set_dir, exist_ok=True)
        for face in ps.faces:
            img_path = os.path.join(VAL_DIR, face.image_file)
            rgb = load_image_rgb(img_path)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bbox = [int(c) for c in face.bbox]
            x1, y1, x2, y2 = bbox
            pad = 15
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(bgr.shape[1], x2 + pad)
            cy2 = min(bgr.shape[0], y2 + pad)
            crop = bgr[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                out_path = os.path.join(set_dir, f"{face.image_file}_{face.face_idx}.jpg")
                cv2.imwrite(out_path, crop)
                crop_count += 1

    print(f"  unknowns/: {len(unknown_sets)} unknown persons, {crop_count} crops")

    # 4. Full results JSON
    json_path = os.path.join(OUTPUT_DIR, "reid_results.json")
    output_json = {
        "config": {
            "embedding_threshold": results["stats"]["embedding_threshold"],
            "n_images": len(image_files),
            "n_students": len(class_list),
        },
        "metrics": {
            "svm_only": {"tp": svm_tp, "fp": svm_fp, "fn": svm_fn,
                         "recognized": len(svm_present)},
            "reid": {"tp": reid_tp, "fp": reid_fp, "fn": reid_fn,
                     "precision": reid_prec, "recall": reid_rec, "f1": reid_f1,
                     "recognized": len(reid_present)},
            "gained_by_reid": sorted(gained),
            "lost_by_reid": sorted(lost),
            "still_missed": missed,
        },
        "person_sets": results["person_sets"],
    }
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2, default=str)
    print(f"  reid_results.json saved")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  SVM-only:  {svm_tp}/{len(gt_present)} present students recognized")
    print(f"  Re-ID:     {reid_tp}/{len(gt_present)} present students recognized")
    improvement = reid_tp - svm_tp
    if improvement > 0:
        print(f"  Re-ID gained {improvement} additional student(s)")
    elif improvement < 0:
        print(f"  Re-ID lost {-improvement} student(s) vs SVM-only")
    else:
        print(f"  Same recognition count (re-ID may still improve grouping)")
    print(f"\n  All outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
