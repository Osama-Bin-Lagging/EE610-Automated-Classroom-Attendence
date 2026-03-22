"""
benchmark_detection.py - Benchmark face detection strategies on classroom validation data.

Tests 10 detection approaches and evaluates attendance accuracy against ground truth.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import json
import argparse
import cv2
import numpy as np
import openpyxl
from PIL import Image, ImageOps
from face_model import FaceRecognitionModel, _get_face_app

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VAL_DIR = os.path.join(PROJECT_DIR, "val_data")
DATASET_DIR = os.path.join(PROJECT_DIR, "course_project_dataset")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "detection_benchmark")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_image_rgb(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def load_ground_truth():
    """Load ground truth from Excel. Returns {dataset_folder_name: 'P'/'A'}."""
    xlsx = os.path.join(VAL_DIR, "Validation_Data.xlsx")
    wb = openpyxl.load_workbook(xlsx)
    ws = wb["Attendance"]
    excel_entries = [(row[0], row[1]) for row in ws.iter_rows(values_only=True)]

    dataset_names = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )

    def normalize(s):
        return set(s.lower().split())

    # Fuzzy match dataset names to Excel names
    gt = {}
    matched_excel = set()
    for dn in dataset_names:
        dn_words = normalize(dn)
        for en, status in excel_entries:
            if en in matched_excel:
                continue
            en_words = normalize(en)
            common = dn_words & en_words
            if dn == en or (len(common) >= 1 and (common == dn_words or common == en_words or len(common) >= 2)):
                gt[dn] = status
                matched_excel.add(en)
                break
        else:
            gt[dn] = "A"  # default absent if no match

    return gt


def iou(box1, box2):
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def center_dist(box1, box2):
    """Euclidean distance between centers of two [x1,y1,x2,y2] boxes."""
    c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) ** 0.5


def dedup_boxes(boxes, min_size_ratio=0.3):
    """Deduplicate overlapping boxes by center proximity. boxes: list of [x1,y1,x2,y2]."""
    if not boxes:
        return []
    # Sort by area descending
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    keep = []
    for box in boxes:
        size = min(box[2] - box[0], box[3] - box[1])
        duplicate = False
        for kept in keep:
            kept_size = min(kept[2] - kept[0], kept[3] - kept[1])
            threshold = min(size, kept_size) * min_size_ratio
            if center_dist(box, kept) < threshold:
                duplicate = True
                break
        if not duplicate:
            keep.append(box)
    return keep


# ── Detector Functions ───────────────────────────────────────────────────────

def detect_retinaface(rgb, det_thresh=0.5):
    """Standard RetinaFace detection."""
    app = _get_face_app()
    old_thresh = app.det_model.det_thresh
    app.det_model.det_thresh = det_thresh
    faces = app.get(rgb)
    app.det_model.det_thresh = old_thresh
    return faces


def detect_retinaface_tiled(rgb, det_thresh=0.1, overlap=400):
    """RetinaFace on 2x2 tiles with dedup."""
    h, w = rgb.shape[:2]
    mid_x = w // 2
    mid_y = h // 2

    tiles = [
        (0, 0, mid_x + overlap, mid_y + overlap),
        (mid_x - overlap, 0, w, mid_y + overlap),
        (0, mid_y - overlap, mid_x + overlap, h),
        (mid_x - overlap, mid_y - overlap, w, h),
    ]

    all_faces = []
    app = _get_face_app()
    old_thresh = app.det_model.det_thresh
    app.det_model.det_thresh = det_thresh

    for tx1, ty1, tx2, ty2 in tiles:
        tile = rgb[ty1:ty2, tx1:tx2]
        faces = app.get(tile)
        for f in faces:
            f.bbox[0] += tx1
            f.bbox[1] += ty1
            f.bbox[2] += tx1
            f.bbox[3] += ty1
            all_faces.append(f)

    app.det_model.det_thresh = det_thresh
    app.det_model.det_thresh = old_thresh

    # Dedup by center proximity
    if not all_faces:
        return []
    boxes = [f.bbox.tolist() for f in all_faces]
    keep_indices = []
    used = set()
    sorted_idx = sorted(range(len(all_faces)),
                         key=lambda i: all_faces[i].det_score, reverse=True)
    for i in sorted_idx:
        if i in used:
            continue
        keep_indices.append(i)
        for j in sorted_idx:
            if j in used or j == i:
                continue
            size_i = min(boxes[i][2]-boxes[i][0], boxes[i][3]-boxes[i][1])
            if center_dist(boxes[i], boxes[j]) < size_i * 0.3:
                used.add(j)

    return [all_faces[i] for i in keep_indices]


def detect_mtcnn(rgb):
    """MTCNN detection via facenet-pytorch."""
    from facenet_pytorch import MTCNN
    import torch
    device = torch.device("cpu")
    mtcnn = MTCNN(keep_all=True, min_face_size=20, device=device,
                  thresholds=[0.6, 0.7, 0.7])
    boxes, probs = mtcnn.detect(rgb)
    if boxes is None:
        return [], []
    results = []
    scores = []
    for box, prob in zip(boxes, probs):
        if prob is not None and prob > 0.5:
            results.append(box.tolist())  # [x1, y1, x2, y2]
            scores.append(float(prob))
    return results, scores


def detect_haar(rgb, aggressive=False, include_profile=False):
    """OpenCV Haar cascade detection."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    frontal_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if aggressive:
        scale_factor = 1.05
        min_neighbors = 3
        min_size = (15, 15)
    else:
        scale_factor = 1.1
        min_neighbors = 5
        min_size = (30, 30)

    frontal = frontal_cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
    )

    boxes = []
    for (x, y, w, h) in (frontal if len(frontal) > 0 else []):
        boxes.append([x, y, x + w, y + h])

    if include_profile:
        profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        profiles = profile_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(15, 15)
        )
        for (x, y, w, h) in (profiles if len(profiles) > 0 else []):
            boxes.append([x, y, x + w, y + h])
        boxes = dedup_boxes(boxes)

    return boxes


def get_embeddings_from_boxes(rgb, boxes, pad_ratio=0.5):
    """Crop each box from image with padding, run RetinaFace to get embedding."""
    app = _get_face_app()
    h, w = rgb.shape[:2]
    results = []  # list of (box, embedding)

    for box in boxes:
        x1, y1, x2, y2 = [int(c) for c in box]
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        crop = rgb[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            continue

        faces = app.get(crop)
        if faces:
            # Pick face closest to center of crop
            crop_cx = (cx2 - cx1) / 2
            crop_cy = (cy2 - cy1) / 2
            best = min(faces, key=lambda f: (
                ((f.bbox[0]+f.bbox[2])/2 - crop_cx)**2 +
                ((f.bbox[1]+f.bbox[3])/2 - crop_cy)**2
            ))
            # Remap bbox to original image coords
            best.bbox[0] += cx1
            best.bbox[1] += cy1
            best.bbox[2] += cx1
            best.bbox[3] += cy1
            results.append(best)

    return results


def detect_haar_then_retinaface(rgb):
    """Haar aggressive → crop+pad → RetinaFace verify."""
    haar_boxes = detect_haar(rgb, aggressive=True)
    return get_embeddings_from_boxes(rgb, haar_boxes, pad_ratio=1.0)


def detect_union(rgb):
    """
    Union of RF(thresh=0.1) global + Haar-only candidates verified by RetinaFace.
    Dedup by center proximity.
    """
    # RetinaFace low threshold
    rf_faces = detect_retinaface(rgb, det_thresh=0.1)
    rf_boxes = [f.bbox.tolist() for f in rf_faces]

    # Haar aggressive
    haar_boxes = detect_haar(rgb, aggressive=True)

    # Find Haar boxes NOT already covered by RF
    haar_only = []
    for hb in haar_boxes:
        covered = False
        for rb in rf_boxes:
            if center_dist(hb, rb) < min(hb[2]-hb[0], hb[3]-hb[1]) * 0.5:
                covered = True
                break
        if not covered:
            haar_only.append(hb)

    # Verify Haar-only with RetinaFace
    haar_verified = get_embeddings_from_boxes(rgb, haar_only, pad_ratio=1.0)

    # Combine
    all_faces = list(rf_faces) + haar_verified

    # Final dedup
    if len(all_faces) <= 1:
        return all_faces
    keep = []
    used = set()
    for i, f in enumerate(all_faces):
        if i in used:
            continue
        keep.append(f)
        for j in range(i+1, len(all_faces)):
            if j in used:
                continue
            size = min(f.bbox[2]-f.bbox[0], f.bbox[3]-f.bbox[1])
            if center_dist(f.bbox.tolist(), all_faces[j].bbox.tolist()) < size * 0.3:
                used.add(j)
    return keep


def detect_tiled_plus_haar_union(rgb):
    """
    Union of RF tiled(thresh=0.1) + Haar-only candidates verified by RetinaFace.
    """
    rf_faces = detect_retinaface_tiled(rgb, det_thresh=0.1)
    rf_boxes = [f.bbox.tolist() for f in rf_faces]

    haar_boxes = detect_haar(rgb, aggressive=True)

    haar_only = []
    for hb in haar_boxes:
        covered = False
        for rb in rf_boxes:
            if center_dist(hb, rb) < min(hb[2]-hb[0], hb[3]-hb[1]) * 0.5:
                covered = True
                break
        if not covered:
            haar_only.append(hb)

    haar_verified = get_embeddings_from_boxes(rgb, haar_only, pad_ratio=1.0)
    all_faces = list(rf_faces) + haar_verified

    if len(all_faces) <= 1:
        return all_faces
    keep = []
    used = set()
    for i, f in enumerate(all_faces):
        if i in used:
            continue
        keep.append(f)
        for j in range(i+1, len(all_faces)):
            if j in used:
                continue
            size = min(f.bbox[2]-f.bbox[0], f.bbox[3]-f.bbox[1])
            if center_dist(f.bbox.tolist(), all_faces[j].bbox.tolist()) < size * 0.3:
                used.add(j)
    return keep


# ── Strategy Registry ────────────────────────────────────────────────────────

STRATEGIES = {
    "RF Default": {
        "desc": "RetinaFace det_size=640, thresh=0.5",
        "has_embeddings": True,
        "fn": lambda rgb: detect_retinaface(rgb, det_thresh=0.5),
    },
    "RF Low-Thresh": {
        "desc": "RetinaFace det_size=640, thresh=0.1",
        "has_embeddings": True,
        "fn": lambda rgb: detect_retinaface(rgb, det_thresh=0.1),
    },
    "RF Tiled": {
        "desc": "RetinaFace 2x2 tiles, overlap=400, thresh=0.1",
        "has_embeddings": True,
        "fn": lambda rgb: detect_retinaface_tiled(rgb, det_thresh=0.1),
    },
    "MTCNN": {
        "desc": "facenet-pytorch MTCNN, min_face=20",
        "has_embeddings": False,
        "fn": lambda rgb: detect_mtcnn(rgb),
    },
    "Haar Default": {
        "desc": "Haar frontal, scale=1.1, minN=5",
        "has_embeddings": False,
        "fn": lambda rgb: detect_haar(rgb, aggressive=False),
    },
    "Haar Aggressive": {
        "desc": "Haar frontal, scale=1.05, minN=3, minSize=15",
        "has_embeddings": False,
        "fn": lambda rgb: detect_haar(rgb, aggressive=True),
    },
    "Haar + Profile": {
        "desc": "Haar frontal aggressive + profile cascade union",
        "has_embeddings": False,
        "fn": lambda rgb: detect_haar(rgb, aggressive=True, include_profile=True),
    },
    "Haar->RF Cascade": {
        "desc": "Haar aggressive -> crop+pad -> RF verify",
        "has_embeddings": True,
        "fn": lambda rgb: detect_haar_then_retinaface(rgb),
    },
    "RF + Haar Union": {
        "desc": "RF(0.1) + Haar-only->RF verify, center dedup",
        "has_embeddings": True,
        "fn": lambda rgb: detect_union(rgb),
    },
    "RF Tiled + Haar": {
        "desc": "RF tiled(0.1) + Haar-only->RF verify, center dedup",
        "has_embeddings": True,
        "fn": lambda rgb: detect_tiled_plus_haar_union(rgb),
    },
}


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_strategy(strategy_name, recognizer, image_paths, ground_truth):
    """
    Run a detection strategy on all images and evaluate attendance accuracy.

    Returns dict with metrics.
    """
    strat = STRATEGIES[strategy_name]
    fn = strat["fn"]
    has_embeddings = strat["has_embeddings"]

    total_faces = 0
    face_counts = []
    predicted_present = set()
    total_time = 0

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        rgb = load_image_rgb(img_path)

        t0 = time.time()
        result = fn(rgb)
        elapsed = time.time() - t0
        total_time += elapsed

        # Handle different return types
        if has_embeddings:
            faces = result
            n_faces = len(faces)
            embeddings = [f.embedding for f in faces]
        else:
            # MTCNN returns (boxes, scores), Haar returns boxes
            if isinstance(result, tuple):
                boxes, scores = result
            else:
                boxes = result
                scores = [1.0] * len(boxes)
            n_faces = len(boxes)

            # Get embeddings by cropping and running RF on each box
            if boxes:
                verified = get_embeddings_from_boxes(rgb, boxes, pad_ratio=0.5)
                embeddings = [f.embedding for f in verified]
            else:
                embeddings = []

        total_faces += n_faces
        face_counts.append(n_faces)

        # Classify each embedding
        for emb in embeddings:
            label, conf = recognizer.predict(emb)
            if label != "Unknown":
                predicted_present.add(label)

        print(f"    {filename}: {n_faces} faces, {len(embeddings)} embeddings ({elapsed:.1f}s)")

    # Compare with ground truth
    gt_present = {name for name, status in ground_truth.items() if status == "P"}
    gt_absent = {name for name, status in ground_truth.items() if status == "A"}

    tp = len(predicted_present & gt_present)   # correctly marked present
    fp = len(predicted_present & gt_absent)    # incorrectly marked present
    fn_count = len(gt_present - predicted_present)  # missed (should be present)
    tn = len(gt_absent - predicted_present)    # correctly marked absent

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn_count) if (tp + fn_count) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "strategy": strategy_name,
        "description": strat["desc"],
        "total_faces": total_faces,
        "avg_faces": total_faces / len(image_paths) if image_paths else 0,
        "face_counts": face_counts,
        "present_predicted": len(predicted_present),
        "tp": tp, "fp": fp, "fn": fn_count, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_time": total_time,
        "avg_time": total_time / len(image_paths) if image_paths else 0,
    }


def save_annotated_sample(strategy_name, recognizer, image_path, output_dir):
    """Save an annotated sample image for a strategy."""
    strat = STRATEGIES[strategy_name]
    fn = strat["fn"]
    has_embeddings = strat["has_embeddings"]

    rgb = load_image_rgb(image_path)
    result = fn(rgb)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    annotated = bgr.copy()

    if has_embeddings:
        faces = result
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            label, conf = recognizer.predict(face.embedding)
            if label != "Unknown":
                color = (0, 200, 0)
                name = label.split("(")[0].strip() if "(" in label else label
                text = f"{name[:15]} ({conf:.0%})"
            else:
                color = (0, 0, 255)
                text = f"? ({conf:.0%})"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, text, (x1, max(y1-5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        if isinstance(result, tuple):
            boxes, _ = result
        else:
            boxes = result
        for box in boxes:
            x1, y1, x2, y2 = [int(c) for c in box]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 165, 0), 2)

    safe_name = strategy_name.replace(" ", "_").replace("+", "plus").replace("->", "to")
    fname = os.path.basename(image_path)
    out_path = os.path.join(output_dir, f"{safe_name}_{fname}")
    cv2.imwrite(out_path, annotated)
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark face detection strategies")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Run only this strategy (by name)")
    parser.add_argument("--images", type=str, default=None,
                        help="Comma-separated image filenames (default: all)")
    parser.add_argument("--annotate", action="store_true",
                        help="Save annotated sample images")
    parser.add_argument("--annotate-image", type=str, default="Image_1.JPG",
                        help="Which image to annotate (default: Image_1.JPG)")
    args = parser.parse_args()

    # Load ground truth
    print("Loading ground truth...")
    gt = load_ground_truth()
    gt_present = sum(1 for s in gt.values() if s == "P")
    gt_absent = sum(1 for s in gt.values() if s == "A")
    print(f"  {len(gt)} students: {gt_present} present, {gt_absent} absent")

    # Load model (use lower threshold for classroom images — enrollment SVM
    # spreads probability across 58 classes so classroom confidences are ~2-4%)
    print("Loading trained model...")
    recognizer = FaceRecognitionModel(threshold=0.02)
    recognizer.load(os.path.join(PROJECT_DIR, "face_database.pkl"))
    recognizer.threshold = 0.02  # override saved threshold

    # Find images
    if args.images:
        image_files = [f.strip() for f in args.images.split(",")]
    else:
        image_files = sorted(
            f for f in os.listdir(VAL_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
    image_paths = [os.path.join(VAL_DIR, f) for f in image_files]
    print(f"  {len(image_paths)} validation images")

    # Select strategies
    if args.strategy:
        strategy_names = [args.strategy]
        if args.strategy not in STRATEGIES:
            print(f"Unknown strategy: {args.strategy}")
            print(f"Available: {', '.join(STRATEGIES.keys())}")
            sys.exit(1)
    else:
        strategy_names = list(STRATEGIES.keys())

    # Run benchmark
    results = []
    for sname in strategy_names:
        print(f"\n{'='*60}")
        print(f"  Strategy: {sname}")
        print(f"  {STRATEGIES[sname]['desc']}")
        print(f"{'='*60}")
        r = evaluate_strategy(sname, recognizer, image_paths, gt)
        results.append(r)
        print(f"  → {r['avg_faces']:.0f} avg faces, "
              f"Recall={r['recall']:.1%}, Precision={r['precision']:.1%}, "
              f"F1={r['f1']:.1%}, Time={r['avg_time']:.1f}s/img")

    # Print comparison table
    print(f"\n{'='*90}")
    print(f"{'Strategy':<22} {'Avg Faces':>9} {'Present':>8} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'Recall':>7} {'Prec':>7} {'F1':>7} {'Time/img':>9}")
    print(f"{'-'*90}")
    for r in results:
        print(f"{r['strategy']:<22} {r['avg_faces']:>9.1f} {r['present_predicted']:>8} "
              f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
              f"{r['recall']:>6.1%} {r['precision']:>6.1%} {r['f1']:>6.1%} "
              f"{r['avg_time']:>8.1f}s")
    print(f"{'='*90}")
    print(f"Ground truth: {gt_present} present, {gt_absent} absent")

    # Save JSON results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, "detection_benchmark.json")
    json_results = []
    for r in results:
        jr = dict(r)
        jr["face_counts"] = r["face_counts"]
        json_results.append(jr)
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Save annotated images
    if args.annotate:
        annot_dir = os.path.join(OUTPUT_DIR, "annotated")
        os.makedirs(annot_dir, exist_ok=True)
        annot_img = os.path.join(VAL_DIR, args.annotate_image)
        print(f"\nSaving annotated images for {args.annotate_image}...")
        for sname in strategy_names:
            out = save_annotated_sample(sname, recognizer, annot_img, annot_dir)
            print(f"  {sname} → {os.path.basename(out)}")

    # Find best strategy
    if len(results) > 1:
        best = max(results, key=lambda r: (r["f1"], r["recall"], -r["avg_time"]))
        print(f"\nBest strategy: {best['strategy']} "
              f"(F1={best['f1']:.1%}, Recall={best['recall']:.1%})")


if __name__ == "__main__":
    main()
