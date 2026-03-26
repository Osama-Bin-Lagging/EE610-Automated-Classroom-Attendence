"""
visualize_reid.py - Streamlit visual tester for cross-image person re-ID.

Run:  streamlit run visualize_reid.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from face_model import FaceRecognitionModel, load_image_rgb
from cache_detections import load_cache
from benchmark_detection import load_ground_truth, VAL_DIR
from reid import PersonReIdentifier, EmbeddingOnlyReIdentifier, cosine_similarity

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Re-ID Visual Tester", layout="wide")
st.title("Cross-Image Person Re-ID — Visual Tester")


# ── Helpers ───────────────────────────────────────────────────────────────────

def crop_face(img_bgr, bbox, pad=15):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(c) for c in bbox]
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = img_bgr[cy1:cy2, cx1:cx2]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.size > 0 else None


def draw_faces(img_bgr, detections, face_labels):
    """Draw bboxes with labels. face_labels: list of (label, color_bgr) per detection."""
    out = img_bgr.copy()
    for det, (label, color) in zip(detections, face_labels):
        bbox = [int(c) for c in det["bbox"]]
        x1, y1, x2, y2 = bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        ty = y1 - 8 if y1 - 8 > 15 else y2 + 18
        cv2.putText(out, label, (x1, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# ── Load data (cached) ───────────────────────────────────────────────────────

@st.cache_resource
def load_data():
    enrollment_embs, val_detections, image_files = load_cache()
    recognizer = FaceRecognitionModel(threshold=0.05)
    recognizer.train(embeddings_dict=enrollment_embs)
    gt = load_ground_truth()
    # Pre-load all val images as BGR
    images_bgr = {}
    for f in image_files:
        rgb = load_image_rgb(os.path.join(VAL_DIR, f))
        images_bgr[f] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return enrollment_embs, val_detections, image_files, recognizer, gt, images_bgr


@st.cache_data
def run_svm_only(_recognizer, _val_detections, image_files):
    """SVM-only predictions per face."""
    results = {}  # {img_file: [(label, conf), ...]}
    present = set()
    for img_file in image_files:
        preds = []
        for det in _val_detections[img_file]:
            label, conf = _recognizer.predict(det["embedding"])
            preds.append((label, conf))
            if label != "Unknown":
                present.add(label)
        results[img_file] = preds
    return results, present


@st.cache_data
def run_reid(_val_detections, _predict_fn, image_files):
    reid = PersonReIdentifier()
    person_sets = reid.process_all(_val_detections, _predict_fn, image_files)
    attendance = reid.get_attendance()
    return person_sets, attendance, reid


enrollment_embs, val_detections, image_files, recognizer, gt, images_bgr = load_data()
svm_results, svm_present = run_svm_only(recognizer, val_detections, image_files)
person_sets, reid_attendance, reid_obj = run_reid(val_detections, recognizer.predict, image_files)

gt_present = {n for n, s in gt.items() if s == "P"}
gt_absent = {n for n, s in gt.items() if s == "A"}
reid_present = set(reid_attendance.keys())

# ── Sidebar metrics ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Metrics")
    svm_tp = len(svm_present & gt_present)
    reid_tp = len(reid_present & gt_present)
    svm_fp = len(svm_present & gt_absent)
    reid_fp = len(reid_present & gt_absent)

    st.metric("Ground Truth Present", len(gt_present))
    c1, c2 = st.columns(2)
    c1.metric("SVM TP", svm_tp)
    c2.metric("Re-ID TP", reid_tp, delta=reid_tp - svm_tp)
    c1.metric("SVM FP", svm_fp)
    c2.metric("Re-ID FP", reid_fp)

    gained = sorted(reid_present - svm_present)
    lost = sorted(svm_present - reid_present)
    if gained:
        st.success(f"Gained: {', '.join(gained)}")
    if lost:
        st.error(f"Lost: {', '.join(lost)}")

    st.divider()
    st.metric("Person Sets", len(person_sets))
    labeled = sum(1 for ps in person_sets if not ps.label.startswith("Unknown Person"))
    st.metric("Labeled / Unknown", f"{labeled} / {len(person_sets) - labeled}")

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_compare, tab_sets, tab_missed, tab_images = st.tabs([
    "SVM vs Re-ID", "Person Sets", "Missed Students", "Browse Images"
])

# ── Tab 1: Side-by-side comparison ───────────────────────────────────────────

with tab_compare:
    st.subheader("SVM-only vs Re-ID — Side by Side")

    # Build face-to-person lookup
    face_lookup = {}
    for ps in person_sets:
        for face in ps.faces:
            face_lookup[(face.image_file, face.face_idx)] = ps

    img_file = st.selectbox("Select image", image_files, key="compare_img")
    bgr = images_bgr[img_file]
    dets = val_detections[img_file]
    svm_preds = svm_results[img_file]

    # SVM-only annotations
    svm_labels = []
    for (label, conf) in svm_preds:
        if label != "Unknown":
            name = label.split("(")[0].strip()[:16]
            svm_labels.append((f"{name} ({conf:.0%})", (0, 200, 0)))
        else:
            svm_labels.append((f"? ({conf:.0%})", (0, 0, 255)))

    # Re-ID annotations
    reid_labels = []
    for fi, det in enumerate(dets):
        ps = face_lookup.get((img_file, fi))
        if ps and not ps.label.startswith("Unknown Person"):
            name = ps.label.split("(")[0].strip()[:16]
            reid_labels.append((f"{name}", (0, 200, 0)))
        elif ps:
            reid_labels.append((f"#{ps.person_id}", (0, 0, 255)))
        else:
            reid_labels.append(("?", (128, 128, 128)))

    col_svm, col_reid = st.columns(2)
    with col_svm:
        st.caption(f"**SVM-only** — {sum(1 for l,_ in svm_labels if l[0]!='?')} recognized")
        st.image(draw_faces(bgr, dets, svm_labels), use_container_width=True)
    with col_reid:
        st.caption(f"**Re-ID** — {sum(1 for l,_ in reid_labels if l[0]!='#' and l[0]!='?')} recognized")
        st.image(draw_faces(bgr, dets, reid_labels), use_container_width=True)

# ── Tab 2: Person Set browser ────────────────────────────────────────────────

with tab_sets:
    st.subheader("Person Sets")

    show = st.radio("Show", ["Labeled only", "Unknown only", "All"],
                    horizontal=True, key="set_filter")

    if show == "Labeled only":
        filtered = [ps for ps in person_sets if not ps.label.startswith("Unknown Person")]
    elif show == "Unknown only":
        filtered = [ps for ps in person_sets if ps.label.startswith("Unknown Person")]
    else:
        filtered = person_sets

    # Sort: labeled first (alphabetical), then unknown by set size desc
    filtered.sort(key=lambda ps: (
        ps.label.startswith("Unknown Person"),
        ps.label if not ps.label.startswith("Unknown Person") else "",
        -len(ps.faces)
    ))

    st.caption(f"Showing {len(filtered)} of {len(person_sets)} person sets")

    for ps in filtered:
        is_labeled = not ps.label.startswith("Unknown Person")
        n_imgs = len(ps.images_covered)
        svm_labels_in_set = [f.svm_label for f in ps.faces if f.svm_label != "Unknown"]
        unique_svm = set(svm_labels_in_set)

        # Header
        if is_labeled:
            gt_status = "P" if ps.label in gt_present else ("A" if ps.label in gt_absent else "?")
            icon = "✅" if gt_status == "P" else "❌"
            header = f"{icon} **{ps.label}** — {len(ps.faces)} faces across {n_imgs} images"
        else:
            header = f"❓ **{ps.label}** — {len(ps.faces)} faces across {n_imgs} images"

        with st.expander(header):
            # Show SVM label info
            if unique_svm:
                st.caption(f"SVM labels in set: {', '.join(sorted(unique_svm))}")
            else:
                st.caption("No SVM labels (all Unknown)")

            # Show face crops in a grid
            cols = st.columns(min(len(ps.faces), 6))
            for i, face in enumerate(ps.faces):
                crop = crop_face(images_bgr[face.image_file], face.bbox)
                if crop is not None:
                    with cols[i % 6]:
                        st.image(crop, caption=f"{face.image_file}\n"
                                 f"SVM: {face.svm_label} ({face.svm_confidence:.1%})\n"
                                 f"det: {face.det_score:.2f}",
                                 width=120)

# ── Tab 3: Missed students deep-dive ─────────────────────────────────────────

with tab_missed:
    st.subheader("Missed Students")
    missed = sorted(gt_present - reid_present)

    if not missed:
        st.success("All present students recognized!")
    else:
        st.warning(f"{len(missed)} students missed: {', '.join(missed)}")

        emb_array = np.array(recognizer.embeddings)
        emb_labels = recognizer.embedding_labels

        selected = st.selectbox("Investigate student", missed, key="missed_sel")

        # Get enrollment embeddings for this student
        student_embs = [emb_array[i] for i, l in enumerate(emb_labels) if l == selected]
        if student_embs:
            centroid = np.mean(student_embs, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

            st.caption(f"Enrollment embeddings: {len(student_embs)}")

            # Find top matches across all detections
            matches = []
            for img_file in image_files:
                for fi, det in enumerate(val_detections[img_file]):
                    sim = cosine_similarity(centroid, det["embedding"])
                    svm_label, svm_conf = svm_results[img_file][fi]
                    ps = face_lookup.get((img_file, fi))
                    matches.append({
                        "img_file": img_file,
                        "face_idx": fi,
                        "similarity": sim,
                        "svm_label": svm_label,
                        "svm_conf": svm_conf,
                        "reid_label": ps.label if ps else "?",
                        "det": det,
                    })

            matches.sort(key=lambda x: x["similarity"], reverse=True)
            top = matches[:12]

            st.write("**Top matches by embedding similarity to enrollment centroid:**")
            cols = st.columns(min(len(top), 6))
            for i, m in enumerate(top):
                crop = crop_face(images_bgr[m["img_file"]], m["det"]["bbox"])
                if crop is not None:
                    with cols[i % 6]:
                        border = "green" if m["similarity"] > 0.4 else "orange" if m["similarity"] > 0.3 else "red"
                        st.image(crop, width=120)
                        st.caption(f"sim={m['similarity']:.3f}\n"
                                   f"{m['img_file']}\n"
                                   f"SVM: {m['svm_label'][:15]}\n"
                                   f"Re-ID: {m['reid_label'][:15]}")

            best_sim = top[0]["similarity"] if top else 0
            if best_sim > 0.35:
                st.info(f"Likely detected but misclassified (best sim={best_sim:.3f}). "
                        f"Top match labeled as '{top[0]['svm_label']}'.")
            else:
                st.warning(f"Likely never detected in any image (best sim={best_sim:.3f}).")

# ── Tab 4: Full image browser ────────────────────────────────────────────────

with tab_images:
    st.subheader("Browse All Images")

    img_file = st.selectbox("Select image", image_files, key="browse_img")
    bgr = images_bgr[img_file]
    dets = val_detections[img_file]

    # Annotate with re-ID labels
    labels = []
    for fi, det in enumerate(dets):
        ps = face_lookup.get((img_file, fi))
        if ps and not ps.label.startswith("Unknown Person"):
            name = ps.label.split("(")[0].strip()[:16]
            labels.append((f"{name}", (0, 200, 0)))
        elif ps:
            labels.append((f"#{ps.person_id}", (0, 0, 255)))
        else:
            labels.append(("?", (128, 128, 128)))

    st.image(draw_faces(bgr, dets, labels), use_container_width=True)

    # Face detail list
    st.caption(f"{len(dets)} faces detected")
    cols = st.columns(min(len(dets), 8))
    for fi, det in enumerate(dets):
        crop = crop_face(bgr, det["bbox"])
        ps = face_lookup.get((img_file, fi))
        svm_label, svm_conf = svm_results[img_file][fi]
        if crop is not None:
            with cols[fi % 8]:
                st.image(crop, width=90)
                reid_lbl = ps.label[:15] if ps else "?"
                st.caption(f"SVM: {svm_label[:12]} ({svm_conf:.0%})\n"
                           f"Re-ID: {reid_lbl}")
