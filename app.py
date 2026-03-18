"""
app.py - Streamlit web UI for the attendance system
"""

import streamlit as st
import os
import json
import tempfile
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from preprocess import preprocess_dataset
from lbph import LBPHFaceRecognizer
from recognize import (
    process_classroom_images,
    save_outputs,
    train_and_save,
    load_image,
)

MODEL_PATH = "face_database.pkl"
PROCESSED_DATASET = "processed_dataset"
CLASS_LIST_PATH = os.path.join(PROCESSED_DATASET, "class_list.json")
OUTPUT_DIR = "outputs"

st.set_page_config(page_title="Attendance System", layout="wide")
st.title("Automated Classroom Attendance System")


def get_recognizer():
    """Load or return cached recognizer."""
    if "recognizer" not in st.session_state:
        st.session_state.recognizer = None
    return st.session_state.recognizer


def get_class_list():
    """Load class list."""
    if os.path.exists(CLASS_LIST_PATH):
        with open(CLASS_LIST_PATH) as f:
            return json.load(f)
    return None


# --- Sidebar ---
with st.sidebar:
    st.header("Setup")

    # Dataset status
    class_list = get_class_list()
    if class_list:
        st.success(f"Dataset loaded: {len(class_list)} students")
    else:
        st.warning("Dataset not preprocessed yet")

    # Model status
    if os.path.exists(MODEL_PATH):
        st.success("Model trained")
        if get_recognizer() is None:
            recognizer = LBPHFaceRecognizer()
            recognizer.load(MODEL_PATH)
            st.session_state.recognizer = recognizer
    else:
        st.warning("Model not trained yet")

    st.divider()

    # Preprocess button
    if st.button("1. Preprocess Dataset", use_container_width=True):
        with st.spinner("Preprocessing images..."):
            stats = preprocess_dataset()
        st.success(f"Done! {stats['success']}/{stats['total']} faces extracted")
        if stats["failed"]:
            st.warning(f"{len(stats['failed'])} images failed")
        st.rerun()

    # Train button
    if st.button("2. Train Model", use_container_width=True):
        if not os.path.exists(CLASS_LIST_PATH):
            st.error("Preprocess dataset first!")
        else:
            with st.spinner("Training LBPH model..."):
                recognizer = train_and_save(PROCESSED_DATASET, MODEL_PATH)
                st.session_state.recognizer = recognizer
            st.success("Model trained!")
            st.rerun()

    st.divider()

    # Threshold slider
    if get_recognizer() is not None:
        threshold = st.slider(
            "Unknown threshold",
            min_value=30.0,
            max_value=80.0,
            value=55.0,
            step=1.0,
            help="Higher = more lenient (fewer unknowns). Lower = stricter.",
        )
        st.session_state.recognizer.threshold = threshold


# --- Main Area ---
recognizer = get_recognizer()
class_list = get_class_list()

if recognizer is None or class_list is None:
    st.info("Use the sidebar to preprocess the dataset and train the model first.")
    st.stop()

# File uploader
uploaded_files = st.file_uploader(
    "Upload classroom images (up to 5)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files and len(uploaded_files) > 5:
    st.error("Maximum 5 images allowed!")
    st.stop()

if uploaded_files:
    # Show uploaded images
    st.subheader("Uploaded Images")
    cols = st.columns(min(len(uploaded_files), 5))
    for i, f in enumerate(uploaded_files):
        with cols[i]:
            st.image(f, caption=f.name, use_container_width=True)

    if st.button("Generate Attendance", type="primary", use_container_width=True):
        # Save uploaded files to temp dir
        temp_paths = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in uploaded_files:
                path = os.path.join(tmpdir, f.name)
                with open(path, "wb") as out:
                    out.write(f.read())
                temp_paths.append(path)

            # Process
            with st.spinner("Detecting and recognizing faces..."):
                attendance, annotated_images, unknown_faces = process_classroom_images(
                    temp_paths, recognizer, class_list
                )
                csv_path = save_outputs(
                    attendance, annotated_images, unknown_faces, class_list
                )

            # Store results in session state
            st.session_state.attendance = attendance
            st.session_state.annotated_images = [
                (name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                for name, img in annotated_images
            ]
            st.session_state.unknown_faces = [
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                for crop, dist in unknown_faces
            ]
            st.session_state.csv_path = csv_path
            st.rerun()

# --- Results ---
if "attendance" in st.session_state:
    st.divider()

    tab1, tab2, tab3 = st.tabs(
        ["Attendance", "Annotated Images", "Unknown Faces"]
    )

    with tab1:
        st.subheader("Attendance Report")
        attendance = st.session_state.attendance
        present = sum(1 for v in attendance.values() if v == "Present")
        absent = sum(1 for v in attendance.values() if v == "Absent")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Students", len(attendance))
        col2.metric("Present", present)
        col3.metric("Absent", absent)

        # Build table
        rows = []
        for i, (name, status) in enumerate(sorted(attendance.items()), 1):
            rows.append({"#": i, "Student Name": name, "Status": status})
        df = pd.DataFrame(rows)

        # Style the table
        def highlight_status(val):
            if val == "Present":
                return "background-color: #d4edda; color: #155724"
            return "background-color: #f8d7da; color: #721c24"

        styled = df.style.applymap(highlight_status, subset=["Status"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Download button
        if os.path.exists(st.session_state.get("csv_path", "")):
            with open(st.session_state.csv_path) as f:
                csv_data = f.read()
            st.download_button(
                "Download Attendance CSV",
                csv_data,
                "attendance.csv",
                "text/csv",
            )

    with tab2:
        st.subheader("Annotated Classroom Images")
        if st.session_state.get("annotated_images"):
            for name, img in st.session_state.annotated_images:
                st.image(img, caption=f"annotated_{name}", use_container_width=True)
        else:
            st.info("No annotated images to show.")

    with tab3:
        st.subheader("Unknown Faces")
        unknowns = st.session_state.get("unknown_faces", [])
        if unknowns:
            cols = st.columns(min(len(unknowns), 5))
            for i, crop in enumerate(unknowns):
                with cols[i % 5]:
                    st.image(crop, caption=f"Unknown {i+1}", width=150)
        else:
            st.info("No unknown faces detected.")
