"""
lbph.py - Local Binary Pattern Histogram face recognition
Implemented from scratch using numpy.
"""

import numpy as np
import cv2
import os
import json
import pickle


def compute_lbp(image):
    """
    Compute Local Binary Pattern for a grayscale image.
    For each pixel, compare with 8 neighbors (clockwise from top-left).
    If neighbor >= center, bit = 1, else bit = 0.
    Returns LBP image (same size, borders are 0).
    """
    h, w = image.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    # 8 neighbor offsets (clockwise from top-left)
    # Bit positions: 7 6 5
    #                0 C 4
    #                1 2 3
    neighbors = [
        (-1, -1, 7), (-1, 0, 6), (-1, 1, 5),
        (0, 1, 4),
        (1, 1, 3), (1, 0, 2), (1, -1, 1),
        (0, -1, 0),
    ]

    center = image[1:-1, 1:-1].astype(np.int16)

    for dy, dx, bit in neighbors:
        neighbor = image[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx].astype(np.int16)
        lbp[1:-1, 1:-1] |= ((neighbor >= center).astype(np.uint8) << bit)

    return lbp


def compute_lbph_features(image, grid_x=8, grid_y=8):
    """
    Compute LBPH feature vector for a grayscale face image.
    1. Compute LBP image
    2. Divide into grid_x * grid_y cells
    3. Compute 256-bin histogram for each cell
    4. Concatenate all histograms into one feature vector
    5. Normalize the feature vector

    Returns: 1D numpy array of length grid_x * grid_y * 256
    """
    lbp = compute_lbp(image)
    h, w = lbp.shape
    cell_h = h // grid_y
    cell_w = w // grid_x

    histograms = []
    for i in range(grid_y):
        for j in range(grid_x):
            # Extract cell
            y1 = i * cell_h
            y2 = (i + 1) * cell_h if i < grid_y - 1 else h
            x1 = j * cell_w
            x2 = (j + 1) * cell_w if j < grid_x - 1 else w
            cell = lbp[y1:y2, x1:x2]

            # Compute histogram (256 bins for 8-bit LBP)
            hist = np.zeros(256, dtype=np.float64)
            for val in cell.ravel():
                hist[val] += 1

            # Normalize histogram
            total = hist.sum()
            if total > 0:
                hist /= total

            histograms.append(hist)

    return np.concatenate(histograms)


def chi_square_distance(h1, h2):
    """
    Chi-square distance between two histograms.
    d = sum((h1 - h2)^2 / (h1 + h2 + eps))
    """
    eps = 1e-10
    diff = h1 - h2
    summ = h1 + h2 + eps
    return np.sum((diff * diff) / summ)


class LBPHFaceRecognizer:
    """LBPH-based face recognizer."""

    def __init__(self, grid_x=8, grid_y=8, threshold=None):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.threshold = threshold
        self.database = {}  # {label: [feature_vectors]}
        self.labels = []  # ordered list of labels

    def train(self, processed_dataset_path):
        """
        Train on preprocessed face crops.
        Expects: processed_dataset_path/{student_name}/face_*.jpg
        """
        self.database = {}
        self.labels = []

        class_list_path = os.path.join(processed_dataset_path, "class_list.json")
        with open(class_list_path) as f:
            class_list = json.load(f)

        for student_name in sorted(class_list.keys()):
            student_path = os.path.join(processed_dataset_path, student_name)
            if not os.path.isdir(student_path):
                continue

            features = []
            for img_file in sorted(os.listdir(student_path)):
                if not img_file.endswith(".jpg"):
                    continue
                img_path = os.path.join(student_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                feat = compute_lbph_features(img, self.grid_x, self.grid_y)
                features.append(feat)

            if features:
                self.database[student_name] = features
                self.labels.append(student_name)

        print(f"Trained on {len(self.labels)} students, "
              f"{sum(len(v) for v in self.database.values())} total face images")

    def predict(self, face_gray):
        """
        Predict identity for a 128x128 grayscale face image.
        Returns: (label, min_distance)
        If min_distance > threshold, label will be "Unknown".
        """
        feat = compute_lbph_features(face_gray, self.grid_x, self.grid_y)

        min_dist = float("inf")
        best_label = "Unknown"

        for label, stored_features in self.database.items():
            for stored_feat in stored_features:
                dist = chi_square_distance(feat, stored_feat)
                if dist < min_dist:
                    min_dist = dist
                    best_label = label

        if self.threshold is not None and min_dist > self.threshold:
            return "Unknown", min_dist

        return best_label, min_dist

    def save(self, path):
        """Save trained model to disk."""
        data = {
            "database": self.database,
            "labels": self.labels,
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
            "threshold": self.threshold,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load trained model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.database = data["database"]
        self.labels = data["labels"]
        self.grid_x = data["grid_x"]
        self.grid_y = data["grid_y"]
        self.threshold = data["threshold"]
        print(f"Model loaded: {len(self.labels)} students")


def leave_one_out_test(processed_dataset_path, grid_x=8, grid_y=8):
    """
    Leave-one-out cross-validation.
    For each student, hold out each image in turn, train on rest, test recognition.
    Returns accuracy and distance statistics for threshold tuning.
    """
    class_list_path = os.path.join(processed_dataset_path, "class_list.json")
    with open(class_list_path) as f:
        class_list = json.load(f)

    # Load all features
    all_data = {}  # {student: [(img_path, feature_vector), ...]}
    for student_name in sorted(class_list.keys()):
        student_path = os.path.join(processed_dataset_path, student_name)
        if not os.path.isdir(student_path):
            continue
        data = []
        for img_file in sorted(os.listdir(student_path)):
            if not img_file.endswith(".jpg"):
                continue
            img_path = os.path.join(student_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            feat = compute_lbph_features(img, grid_x, grid_y)
            data.append((img_file, feat))
        if data:
            all_data[student_name] = data

    correct = 0
    total = 0
    same_person_dists = []
    diff_person_dists = []

    for test_student in all_data:
        for i, (test_file, test_feat) in enumerate(all_data[test_student]):
            min_dist = float("inf")
            best_label = None

            for train_student in all_data:
                for j, (train_file, train_feat) in enumerate(all_data[train_student]):
                    # Skip the held-out image
                    if train_student == test_student and j == i:
                        continue

                    dist = chi_square_distance(test_feat, train_feat)

                    if train_student == test_student:
                        same_person_dists.append(dist)
                    else:
                        diff_person_dists.append(dist)

                    if dist < min_dist:
                        min_dist = dist
                        best_label = train_student

            if best_label == test_student:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    same_arr = np.array(same_person_dists)
    diff_arr = np.array(diff_person_dists)

    print(f"\n{'='*50}")
    print(f"Leave-One-Out Results:")
    print(f"  Accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"\n  Same-person distances:")
    print(f"    Mean: {same_arr.mean():.4f}, Std: {same_arr.std():.4f}")
    print(f"    Min:  {same_arr.min():.4f}, Max: {same_arr.max():.4f}")
    print(f"    Median: {np.median(same_arr):.4f}")
    print(f"\n  Different-person distances:")
    print(f"    Mean: {diff_arr.mean():.4f}, Std: {diff_arr.std():.4f}")
    print(f"    Min:  {diff_arr.min():.4f}, Max: {diff_arr.max():.4f}")
    print(f"    Median: {np.median(diff_arr):.4f}")

    # Suggest threshold (midpoint between max same-person and min diff-person)
    suggested = (same_arr.max() + diff_arr.min()) / 2
    # Or use percentile-based approach
    p95_same = np.percentile(same_arr, 95)
    p5_diff = np.percentile(diff_arr, 5)
    suggested2 = (p95_same + p5_diff) / 2
    print(f"\n  Suggested threshold (max-same/min-diff midpoint): {suggested:.4f}")
    print(f"  Suggested threshold (95th/5th percentile midpoint): {suggested2:.4f}")

    return accuracy, suggested2


if __name__ == "__main__":
    print("Running leave-one-out test...")
    accuracy, threshold = leave_one_out_test("processed_dataset")
