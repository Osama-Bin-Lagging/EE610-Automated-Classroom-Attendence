"""
app.py - Streamlit web UI for automated classroom attendance.
Accepts images, videos, and ZIP files. Uses the production pipeline from run.py.
"""

import streamlit as st
import os
import json
import csv
import tempfile
import zipfile
import cv2
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Classroom Attendance", layout="wide")
st.title("Automated Classroom Attendance System")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
TEMPLATE_PATH = os.path.join(PROJECT_DIR, "attendance_output.xlsx")
TEMPLATE_NAMES = [
    "Sripradheep K", "Aditya Kumar Roy", "Anupam Sardar", "Devesh Soni",
    "Jadhav Shubham Sudhakar", "Kamalesh Barman", "Kushagra Gehlot",
    "Lohitaksh Mahajan", "Pulkit", "Terli Tejaswi", "Vedagya Bhalotia",
    "Jha Ayush", "Tarshit Sehgal", "Vishwam Hemang Patel",
    "Moon Aman Milind", "K. P. Lakshmeesh", "Nistala Sreechandana",
    "Pyla Ramya", "Shrutija Swain", "Srushti Rajwade",
    "Atharva Uday Jaltare", "Parul Diwan", "Ajinkya",
    "Ajinkya Prashant Pawar", "Devtanu Barman",
    "Gogineni Venkat Sumanth", "Shaik Rehna Afroz", "Rishabh Bhardwaj",
    "Praveen Vijayarajasekharan", "Saurabh Gupta", "Gurbani Jeet Sunil",
    "Shreeram Anil Jadhav", "Shaurya Goyal", "Aboli Ganesh Malshikare",
    "Shreya Nigam", "Baral Prathmesh Santosh", "Krunal Vaghela",
    "Manit Jhajharia", "Foram Payal Trivedi", "Aman Russell",
    "Arjun Singh", "Soumya Gour", "Jadhav Jay Karbhari",
    "Mihir Gajanan Wani", "Yashasvee Vijay Taiwade", "Manaswi Goyal",
    "Ayush Ashish Kumar", "Chhavi Yadav", "Deepak Kumar", "Divig Bansal",
    "Shah Pratham Manish", "Sameer Anand Jha", "Nithin Jonnalagadda",
    "Ganesh Dattu Yadawate", "Jyoti", "Rownak Tiwari", "Dharmveer",
    "Rajeswar Banerjee", "Akarsh Saxena", "Mithilesh Kumar Verma",
    "Ankit Raj", "Abhishek", "Samyak Sanjay Parakh", "Santosh Singh",
    "Ayush Singh", "Suraj Adhikari", "Manas Sudam Patil",
    "Maitreya Gautam Shelare", "Anish Prashant Mayanache",
    "Amresh Kumar Jha",
]
OUTPUT_XLSX = os.path.join(PROJECT_DIR, "EE610_Project_Output.xlsx")


# Pipeline name -> Template name (only for mismatched names; the other 48 match exactly)
_PIPELINE_TO_TEMPLATE = {
    "Abhishek Abhishek": "Abhishek",
    "Amresh Jha": "Amresh Kumar Jha",
    "Anish Mayanache": "Anish Prashant Mayanache",
    "Ganesh Yadawate": "Ganesh Dattu Yadawate",
    "Jyoti Jyoti": "Jyoti",
    "Maitreya Shelare": "Maitreya Gautam Shelare",
    "Manas Patil": "Manas Sudam Patil",
    "Pratham Manish Shah": "Shah Pratham Manish",
    "Pulkit Pulkit": "Pulkit",
    "Samyak Parakh": "Samyak Sanjay Parakh",
}
_TEMPLATE_TO_PIPELINE = {v: k for k, v in _PIPELINE_TO_TEMPLATE.items()}


def update_output_xlsx(attendance_csv_path, test_name):
    """Write pipeline results as a new column in the output xlsx."""
    if os.path.exists(OUTPUT_XLSX):
        df = pd.read_excel(OUTPUT_XLSX)
    else:
        df = pd.DataFrame({"Name": TEMPLATE_NAMES})

    att = pd.read_csv(attendance_csv_path)
    status_map = dict(zip(att["Name"], att["Status"]))

    col_values = []
    for tname in df["Name"]:
        # Map template name to pipeline name (direct or via mapping)
        pname = _TEMPLATE_TO_PIPELINE.get(tname, tname)
        if status_map.get(pname) == "Present":
            col_values.append("P")
        else:
            col_values.append("A")

    df[test_name] = col_values
    df.to_excel(OUTPUT_XLSX, index=False)
    return OUTPUT_XLSX


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")

    # Model status
    from run import load_model, MODEL_PATH
    if os.path.exists(MODEL_PATH):
        st.success("Model loaded (ArcFace + SVM, 2x aug)")
    else:
        st.warning("No model found — will train on first run")

    test_name = st.text_input("Test name (column in output xlsx)",
                              value="Test 1 (A/P)",
                              help="Each run adds a column with this name to EE610_Project_Output.xlsx")

    fps = st.slider("Video frame extraction rate", 0.5, 5.0, 1.0, 0.5,
                     help="Frames per second to extract from video files")

    st.divider()
    st.markdown("""
    **Accepted inputs:**
    - Images: JPG, JPEG, PNG
    - Videos: MP4, AVI, MOV, MKV
    - ZIP: containing any of the above

    Mix and match — upload any combination.
    """)


# ── Input Processing Helpers ─────────────────────────────────────────────────

def extract_frames(video_path, output_dir, target_fps):
    """Extract frames from a video at the given fps. Returns list of image paths."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30
    frame_interval = max(1, int(video_fps / target_fps))
    paths = []
    frame_idx = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            fname = f"{video_name}_frame_{frame_idx:06d}.jpg"
            fpath = os.path.join(output_dir, fname)
            cv2.imwrite(fpath, frame)
            paths.append(fpath)
        frame_idx += 1
    cap.release()
    return paths


def process_zip(zip_path, output_dir, target_fps):
    """Extract a ZIP, find images and videos, return list of image paths."""
    paths = []
    extract_dir = os.path.join(output_dir, "zip_contents")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    for root, _, files in os.walk(extract_dir):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            fpath = os.path.join(root, f)
            if ext in IMAGE_EXTS:
                paths.append(fpath)
            elif ext in VIDEO_EXTS:
                paths.extend(extract_frames(fpath, output_dir, target_fps))
    return paths


def resolve_uploads(uploaded_files, work_dir, target_fps):
    """Resolve all uploaded files to a flat list of image paths.
    Returns (image_paths, source_stats)."""
    image_paths = []
    stats = {"images": 0, "video_frames": 0, "zip_files": 0}
    for uf in uploaded_files:
        ext = os.path.splitext(uf.name)[1].lower()
        saved = os.path.join(work_dir, uf.name)
        with open(saved, "wb") as f:
            f.write(uf.read())
        if ext in IMAGE_EXTS:
            image_paths.append(saved)
            stats["images"] += 1
        elif ext in VIDEO_EXTS:
            frames = extract_frames(saved, work_dir, target_fps)
            image_paths.extend(frames)
            stats["video_frames"] += len(frames)
        elif ext == ".zip":
            zip_images = process_zip(saved, work_dir, target_fps)
            image_paths.extend(zip_images)
            stats["zip_files"] += 1
    return image_paths, stats


# ── Main Area ────────────────────────────────────────────────────────────────

uploaded_files = st.file_uploader(
    "Upload classroom images, videos, or ZIP files",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv", "zip"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.caption(f"{len(uploaded_files)} file(s) selected")

    if st.button("Generate Attendance", type="primary", width="stretch"):
        with tempfile.TemporaryDirectory() as work_dir:
            # Resolve all inputs to images
            with st.spinner("Processing uploads..."):
                image_paths, source_stats = resolve_uploads(uploaded_files, work_dir, fps)

            if not image_paths:
                st.error("No images found in the uploaded files.")
                st.stop()

            n_total = source_stats["images"] + source_stats["video_frames"]
            parts = []
            if source_stats["images"]:
                parts.append(f"{source_stats['images']} images")
            if source_stats["video_frames"]:
                parts.append(f"{source_stats['video_frames']} video frames")
            if source_stats["zip_files"]:
                parts.append(f"{source_stats['zip_files']} ZIP(s)")
            st.info(f"Processing {len(image_paths)} images from: {', '.join(parts)}")

            # Run pipeline
            with st.spinner("Running attendance pipeline (detection → re-ID → output)..."):
                from run import run_pipeline
                run_dir = run_pipeline(image_paths)

            # Write results to output xlsx
            csv_path = os.path.join(run_dir, "attendance.csv")
            xlsx_path = update_output_xlsx(csv_path, test_name)

            st.session_state.run_dir = run_dir
            st.session_state.source_stats = source_stats
            st.session_state.xlsx_path = xlsx_path
            st.rerun()

# ── Results Display ──────────────────────────────────────────────────────────

if "run_dir" in st.session_state:
    run_dir = st.session_state.run_dir
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Attendance", "Annotated Images", "Unknown Faces", "Summary"]
    )

    # Tab 1: Attendance
    with tab1:
        csv_path = os.path.join(run_dir, "attendance.csv")
        if os.path.exists(csv_path):
            rows = []
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    rows.append(row)

            present = sum(1 for r in rows if r["Status"] == "Present")
            absent = sum(1 for r in rows if r["Status"] == "Absent")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Students", len(rows))
            col2.metric("Present", present)
            col3.metric("Absent", absent)

            df = pd.DataFrame(rows)
            df.index = range(1, len(df) + 1)
            df.index.name = "#"

            def highlight_status(val):
                if val == "Present":
                    return "background-color: #d4edda; color: #155724"
                return "background-color: #f8d7da; color: #721c24"

            styled = df.style.map(highlight_status, subset=["Status"])
            st.dataframe(styled, width="stretch")

            col_a, col_b = st.columns(2)
            with col_a:
                with open(csv_path) as f:
                    st.download_button("Download CSV", f.read(), "attendance.csv", "text/csv")
            with col_b:
                if os.path.exists(OUTPUT_XLSX):
                    with open(OUTPUT_XLSX, "rb") as f:
                        st.download_button("Download Output XLSX", f.read(),
                                           "EE610_Project_Output.xlsx",
                                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Tab 2: Annotated Images
    with tab2:
        ann_dir = os.path.join(run_dir, "annotated")
        if os.path.isdir(ann_dir):
            ann_files = sorted(f for f in os.listdir(ann_dir)
                               if f.lower().endswith((".jpg", ".jpeg", ".png")))
            if ann_files:
                for fname in ann_files:
                    img = Image.open(os.path.join(ann_dir, fname))
                    st.image(img, caption=fname, width="stretch")
            else:
                st.info("No annotated images.")
        else:
            st.info("No annotated images directory found.")

    # Tab 3: Unknown Faces
    with tab3:
        unk_dir = os.path.join(run_dir, "unknowns")
        if os.path.isdir(unk_dir):
            person_dirs = sorted(d for d in os.listdir(unk_dir)
                                 if os.path.isdir(os.path.join(unk_dir, d)))
            if person_dirs:
                for person in person_dirs:
                    pdir = os.path.join(unk_dir, person)
                    crops = sorted(f for f in os.listdir(pdir)
                                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
                    if not crops:
                        continue
                    label = person.replace("_", " ")
                    st.markdown(f"**{label}** — {len(crops)} detection(s)")
                    cols = st.columns(min(len(crops), 6))
                    for i, crop_name in enumerate(crops):
                        with cols[i % 6]:
                            img = Image.open(os.path.join(pdir, crop_name))
                            st.image(img, caption=crop_name, width=120)
                    st.divider()
            else:
                st.info("No unknown faces detected.")
        else:
            st.info("No unknowns directory found.")

    # Tab 4: Summary
    with tab4:
        summary_path = os.path.join(run_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)

            # Config
            st.subheader("Pipeline Configuration")
            config = summary.get("config", {})
            cfg_cols = st.columns(4)
            cfg_cols[0].metric("SVM Threshold", config.get("threshold", "—"))
            cfg_cols[1].metric("Det Score Min", config.get("det_score_min", "—"))
            cfg_cols[2].metric("Augmentation", f"{config.get('n_augmentation', '—')}x")
            cfg_cols[3].metric("Enrolled Students", config.get("n_enrolled_students", "—"))

            # Input source breakdown
            source_stats = st.session_state.get("source_stats", {})
            if source_stats:
                st.subheader("Input Sources")
                src_cols = st.columns(3)
                src_cols[0].metric("Direct Images", source_stats.get("images", 0))
                src_cols[1].metric("Video Frames", source_stats.get("video_frames", 0))
                src_cols[2].metric("ZIP Files", source_stats.get("zip_files", 0))

            # Totals
            st.subheader("Results")
            totals = summary.get("totals", {})
            tot_cols = st.columns(4)
            tot_cols[0].metric("Faces Detected", totals.get("faces_detected", "—"))
            tot_cols[1].metric("Students Present", totals.get("students_present", "—"))
            tot_cols[2].metric("Students Absent", totals.get("students_absent", "—"))
            tot_cols[3].metric("Unknown Groups", totals.get("unknown_person_groups", "—"))

            # Per-image table
            per_image = summary.get("per_image", {})
            if per_image:
                st.subheader("Per-Image Breakdown")
                pi_rows = []
                for img, data in sorted(per_image.items()):
                    pi_rows.append({
                        "Image": img,
                        "Faces Detected": data.get("faces_detected", 0),
                        "Recognized": data.get("recognized", 0),
                    })
                st.dataframe(pd.DataFrame(pi_rows), width="stretch", hide_index=True)

            # Re-ID stats
            reid_stats = summary.get("reid_stats", {})
            if reid_stats:
                st.subheader("Re-ID Statistics")
                ri_cols = st.columns(3)
                ri_cols[0].metric("Total Person Sets", reid_stats.get("total_person_sets", "—"))
                ri_cols[1].metric("Labeled", reid_stats.get("labeled", "—"))
                ri_cols[2].metric("Unlabeled", reid_stats.get("unlabeled", "—"))
        else:
            st.info("No summary file found.")
