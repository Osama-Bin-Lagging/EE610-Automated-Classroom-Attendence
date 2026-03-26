"""
reid.py - Cross-image person re-identification module.

Matches faces across multiple classroom images to propagate known identities
to unknown detections of the same person. Uses embedding similarity, spatial
anchors, relative position descriptors, and Hungarian assignment.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import fcluster, linkage
import cv2


@dataclass
class FaceDetection:
    """A single detected face in one image."""
    image_idx: int          # which image this face came from
    image_file: str
    bbox: List[float]       # [x1, y1, x2, y2]
    embedding: np.ndarray   # 512-d ArcFace embedding
    det_score: float
    kps: Optional[np.ndarray]  # facial keypoints
    svm_label: str          # SVM prediction ("Unknown" or student name)
    svm_confidence: float
    face_idx: int = 0       # index within its image

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2)

    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass
class PersonSet:
    """A group of faces believed to be the same physical person."""
    faces: List[FaceDetection] = field(default_factory=list)
    label: Optional[str] = None
    person_id: int = 0

    @property
    def mean_embedding(self) -> np.ndarray:
        embs = np.array([f.embedding for f in self.faces])
        mean = embs.mean(axis=0)
        return mean / (np.linalg.norm(mean) + 1e-8)

    @property
    def images_covered(self) -> set:
        return {f.image_idx for f in self.faces}

    @property
    def svm_labels(self) -> List[str]:
        return [f.svm_label for f in self.faces if f.svm_label != "Unknown"]

    def add_face(self, face: FaceDetection):
        self.faces.append(face)

    def has_image(self, image_idx: int) -> bool:
        return image_idx in self.images_covered


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors (0 = identical, 2 = opposite)."""
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return 1.0 - sim


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ── Method A: Embedding Clustering ────────────────────────────────────────────

def embedding_distance_matrix(faces: List[FaceDetection],
                              person_sets: List[PersonSet]) -> np.ndarray:
    """
    Compute cosine distance from each face to each PersonSet's mean embedding.
    Returns (n_faces, n_sets) distance matrix.
    """
    if not faces or not person_sets:
        return np.empty((len(faces), len(person_sets)))

    face_embs = np.array([f.embedding for f in faces])
    set_embs = np.array([ps.mean_embedding for ps in person_sets])

    # Normalize
    face_norms = np.linalg.norm(face_embs, axis=1, keepdims=True) + 1e-8
    set_norms = np.linalg.norm(set_embs, axis=1, keepdims=True) + 1e-8
    face_embs = face_embs / face_norms
    set_embs = set_embs / set_norms

    return cdist(face_embs, set_embs, metric="cosine")


def auto_threshold(distances: np.ndarray, default: float = 0.6) -> float:
    """
    Find optimal distance threshold from gap in sorted distances.
    Looks for the largest gap between consecutive sorted distances.
    """
    flat = distances.flatten()
    flat = flat[flat < 1.5]  # ignore very dissimilar
    if len(flat) < 4:
        return default

    sorted_d = np.sort(flat)
    gaps = np.diff(sorted_d)

    # Only look in the middle range (0.2 to 0.9) for meaningful gaps
    valid = (sorted_d[:-1] > 0.15) & (sorted_d[:-1] < 0.9)
    if not valid.any():
        return default

    gap_idx = np.where(valid)[0]
    if len(gap_idx) == 0:
        return default

    best = gap_idx[np.argmax(gaps[gap_idx])]
    threshold = (sorted_d[best] + sorted_d[best + 1]) / 2

    return np.clip(threshold, 0.3, 0.8)


# ── Method B: Anchor-Based Spatial Mapping ────────────────────────────────────

def find_anchors(faces_new: List[FaceDetection],
                 faces_prev: List[FaceDetection]) -> List[Tuple[FaceDetection, FaceDetection]]:
    """
    Find anchor pairs: faces recognized as the same person in both images.
    Only uses faces with SVM labels (not "Unknown").
    """
    anchors = []
    prev_by_label = {}
    for f in faces_prev:
        if f.svm_label != "Unknown":
            prev_by_label.setdefault(f.svm_label, []).append(f)

    for f_new in faces_new:
        if f_new.svm_label != "Unknown" and f_new.svm_label in prev_by_label:
            # Pick closest by embedding
            best = min(prev_by_label[f_new.svm_label],
                       key=lambda fp: cosine_distance(f_new.embedding, fp.embedding))
            anchors.append((f_new, best))
    return anchors


def spatial_mapping_cost(faces_new: List[FaceDetection],
                         person_sets: List[PersonSet],
                         image_idx: int,
                         all_faces_by_image: Dict[int, List[FaceDetection]]) -> Optional[np.ndarray]:
    """
    Method B: Use anchor-based affine transform to predict face positions.
    Returns (n_faces, n_sets) distance matrix or None if insufficient anchors.
    """
    # Collect all previous faces from images covered by the person sets
    prev_images = set()
    for ps in person_sets:
        prev_images.update(ps.images_covered)

    if not prev_images:
        return None

    # Try each previous image to find anchors
    best_cost = None
    for prev_img_idx in prev_images:
        prev_faces = all_faces_by_image.get(prev_img_idx, [])
        if not prev_faces:
            continue

        anchors = find_anchors(faces_new, prev_faces)
        if len(anchors) < 2:
            continue

        # Build affine transform from anchor positions
        src_pts = np.array([a[0].center for a in anchors], dtype=np.float32)
        dst_pts = np.array([a[1].center for a in anchors], dtype=np.float32)

        if len(anchors) >= 3:
            M, inliers = cv2.estimateAffine2D(src_pts.reshape(-1, 1, 2),
                                                dst_pts.reshape(-1, 1, 2),
                                                method=cv2.RANSAC)
        else:
            # 2 anchors: use partial affine (translation + scale + rotation)
            M, inliers = cv2.estimateAffinePartial2D(
                src_pts.reshape(-1, 1, 2),
                dst_pts.reshape(-1, 1, 2))

        if M is None:
            continue

        # Project new face positions to previous image space
        new_centers = np.array([f.center for f in faces_new], dtype=np.float32)
        projected = cv2.transform(new_centers.reshape(-1, 1, 2), M).reshape(-1, 2)

        # Compute distance to each PersonSet's face in prev_img
        cost = np.full((len(faces_new), len(person_sets)), 1e6)
        for j, ps in enumerate(person_sets):
            ps_faces_in_prev = [f for f in ps.faces if f.image_idx == prev_img_idx]
            if not ps_faces_in_prev:
                continue
            for pf in ps_faces_in_prev:
                pc = np.array(pf.center)
                # Normalize by image diagonal
                img_diag = max(1.0, np.sqrt(
                    (max(f.bbox[2] for f in prev_faces) - min(f.bbox[0] for f in prev_faces))**2 +
                    (max(f.bbox[3] for f in prev_faces) - min(f.bbox[1] for f in prev_faces))**2
                ))
                dists = np.linalg.norm(projected - pc, axis=1) / img_diag
                cost[:, j] = np.minimum(cost[:, j], dists)

        if best_cost is None or cost.min() < best_cost.min():
            best_cost = cost

    return best_cost


# ── Method C: Relative Position Encoding ──────────────────────────────────────

def relative_position_descriptor(face: FaceDetection,
                                 all_faces: List[FaceDetection],
                                 k: int = 5) -> np.ndarray:
    """
    Build rotation-invariant spatial descriptor for a face.
    Uses normalized distances to K nearest neighbors, pairwise angle diffs,
    and neighbor embedding similarities.
    """
    if len(all_faces) <= 1:
        return np.zeros(k * 3)

    center = np.array(face.center)
    others = [f for f in all_faces if f is not face]
    if not others:
        return np.zeros(k * 3)

    # Sort by distance
    other_centers = np.array([f.center for f in others])
    dists = np.linalg.norm(other_centers - center, axis=1)
    sort_idx = np.argsort(dists)
    k_actual = min(k, len(others))

    # Normalize distances by median distance (scale-invariant)
    median_dist = np.median(dists) + 1e-8
    norm_dists = dists[sort_idx[:k_actual]] / median_dist

    # Pairwise angle differences (rotation-invariant)
    angles = np.arctan2(other_centers[sort_idx[:k_actual], 1] - center[1],
                        other_centers[sort_idx[:k_actual], 0] - center[0])
    angle_diffs = np.diff(angles) if len(angles) > 1 else np.array([0.0])
    angle_diffs = np.mod(angle_diffs + np.pi, 2 * np.pi) - np.pi  # wrap to [-pi, pi]

    # Embedding similarities to neighbors
    emb_sims = np.array([cosine_similarity(face.embedding, others[i].embedding)
                         for i in sort_idx[:k_actual]])

    # Pad to fixed size
    desc = np.zeros(k * 3)
    desc[:k_actual] = norm_dists
    desc[k:k + min(k, len(angle_diffs))] = angle_diffs[:k]
    desc[2*k:2*k + k_actual] = emb_sims

    return desc


def spatial_context_distance(face: FaceDetection,
                             person_set: PersonSet,
                             faces_in_new_img: List[FaceDetection],
                             all_faces_by_image: Dict[int, List[FaceDetection]],
                             k: int = 5) -> float:
    """
    Method C: Distance between spatial descriptors of a face and a PersonSet member.
    """
    desc_new = relative_position_descriptor(face, faces_in_new_img, k)

    min_dist = float("inf")
    for pf in person_set.faces:
        img_faces = all_faces_by_image.get(pf.image_idx, [])
        desc_ps = relative_position_descriptor(pf, img_faces, k)
        dist = np.linalg.norm(desc_new - desc_ps)
        min_dist = min(min_dist, dist)

    return min_dist


# ── Method D: Hungarian Optimal Assignment ────────────────────────────────────

def hungarian_assignment(cost_matrix: np.ndarray,
                         max_cost: float = 1.5) -> List[Tuple[int, int]]:
    """
    Optimal 1-to-1 assignment using Hungarian algorithm.
    Returns list of (face_idx, set_idx) pairs with cost below max_cost.
    """
    if cost_matrix.size == 0:
        return []

    # Pad to square if needed
    n_faces, n_sets = cost_matrix.shape
    max_dim = max(n_faces, n_sets)
    padded = np.full((max_dim, max_dim), max_cost * 2)
    padded[:n_faces, :n_sets] = cost_matrix

    row_ind, col_ind = linear_sum_assignment(padded)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < n_faces and c < n_sets and cost_matrix[r, c] < max_cost:
            matches.append((r, c))
    return matches


# ── Weight Learning ───────────────────────────────────────────────────────────

def learn_weights_from_anchors(faces_new: List[FaceDetection],
                               person_sets: List[PersonSet],
                               faces_in_new_img: List[FaceDetection],
                               all_faces_by_image: Dict[int, List[FaceDetection]],
                               image_idx: int) -> Tuple[float, float]:
    """
    Learn alpha (embedding weight) and beta (spatial weight) from anchor pairs.
    Falls back to defaults if insufficient anchors.
    """
    alpha_default, beta_default = 0.7, 0.3

    # Find recognized faces that match existing sets
    anchor_emb_dists = []
    anchor_spat_dists = []

    for face in faces_new:
        if face.svm_label == "Unknown":
            continue
        for j, ps in enumerate(person_sets):
            if ps.label == face.svm_label:
                emb_d = cosine_distance(face.embedding, ps.mean_embedding)
                spat_d = spatial_context_distance(
                    face, ps, faces_in_new_img, all_faces_by_image)
                anchor_emb_dists.append(emb_d)
                anchor_spat_dists.append(spat_d)

    if len(anchor_emb_dists) < 2:
        return alpha_default, beta_default

    # Weight inversely by variance (more reliable signal gets more weight)
    var_emb = np.var(anchor_emb_dists) + 1e-8
    var_spat = np.var(anchor_spat_dists) + 1e-8
    w_emb = 1.0 / var_emb
    w_spat = 1.0 / var_spat
    total = w_emb + w_spat
    alpha = w_emb / total
    beta = w_spat / total

    return float(alpha), float(beta)


# ── PersonReIdentifier ───────────────────────────────────────────────────────

class PersonReIdentifier:
    """
    Incremental person re-identification across multiple images.

    Processes images one at a time (most-recognized first), building PersonSets —
    groups of faces believed to be the same physical person. Uses embedding
    similarity, anchor-based spatial mapping, relative position descriptors,
    and Hungarian assignment.
    """

    def __init__(self, embedding_threshold: float = None,
                 weight_embedding: float = 0.5,
                 weight_spatial: float = 0.2,
                 weight_hungarian: float = 0.3,
                 merge_threshold: float = 0.4):
        self.embedding_threshold = embedding_threshold  # None = auto
        self.w_emb = weight_embedding
        self.w_spat = weight_spatial
        self.w_hung = weight_hungarian
        self.merge_threshold = merge_threshold

        self.person_sets: List[PersonSet] = []
        self.all_faces: List[FaceDetection] = []
        self.all_faces_by_image: Dict[int, List[FaceDetection]] = {}
        self._next_person_id = 1

    def _new_person_id(self) -> int:
        pid = self._next_person_id
        self._next_person_id += 1
        return pid

    def process_all(self, detections_by_image: Dict[str, list],
                    predict_fn, image_files: List[str] = None) -> List[PersonSet]:
        """
        Main entry point: process all images and return labeled PersonSets.

        Args:
            detections_by_image: {img_file: [{bbox, embedding, det_score, kps}, ...]}
            predict_fn: callable(embedding) -> (label, confidence)
            image_files: optional ordering (default: sorted by recognition count desc)
        """
        # Step 1: Run SVM on all faces, build FaceDetection objects
        faces_by_image: Dict[int, List[FaceDetection]] = {}
        file_to_idx = {}

        if image_files is None:
            image_files = sorted(detections_by_image.keys())

        for idx, img_file in enumerate(image_files):
            file_to_idx[img_file] = idx
            faces = []
            for fi, det in enumerate(detections_by_image[img_file]):
                label, conf = predict_fn(det["embedding"])
                fd = FaceDetection(
                    image_idx=idx,
                    image_file=img_file,
                    bbox=det["bbox"],
                    embedding=det["embedding"],
                    det_score=det["det_score"],
                    kps=np.array(det["kps"]) if det.get("kps") else None,
                    svm_label=label,
                    svm_confidence=conf,
                    face_idx=fi,
                )
                faces.append(fd)
                self.all_faces.append(fd)
            faces_by_image[idx] = faces
            self.all_faces_by_image[idx] = faces

        # Step 2: Sort images by recognition count (descending) → process most informative first
        def recognition_count(idx):
            return sum(1 for f in faces_by_image[idx] if f.svm_label != "Unknown")

        image_order = sorted(faces_by_image.keys(), key=recognition_count, reverse=True)

        # Step 3: Incremental set building
        for img_idx in image_order:
            self._process_image(img_idx, faces_by_image[img_idx])

        # Step 4: Post-processing
        self._merge_similar_sets()
        self._enforce_one_per_image()
        self._propagate_labels()

        return self.person_sets

    def _process_image(self, image_idx: int, faces: List[FaceDetection]):
        """Process one image: match faces to existing sets or create new ones."""
        if not faces:
            return

        # Filter sets that don't already have a face from this image
        available_sets = [ps for ps in self.person_sets
                          if not ps.has_image(image_idx)]

        if not available_sets:
            # No existing sets — create new PersonSet for each face
            for face in faces:
                ps = PersonSet(faces=[face], person_id=self._new_person_id())
                if face.svm_label != "Unknown":
                    ps.label = face.svm_label
                self.person_sets.append(ps)
            return

        # Build combined cost matrix
        n_faces = len(faces)
        n_sets = len(available_sets)

        # Method A: Embedding distance
        emb_cost = embedding_distance_matrix(faces, available_sets)

        # Auto-threshold if needed
        if self.embedding_threshold is None:
            self.embedding_threshold = auto_threshold(emb_cost)

        # Method B: Spatial mapping (when anchors available)
        spat_map_cost = spatial_mapping_cost(
            faces, available_sets, image_idx, self.all_faces_by_image)

        # Method C: Relative position context
        spat_ctx_cost = np.zeros((n_faces, n_sets))
        for i, face in enumerate(faces):
            for j, ps in enumerate(available_sets):
                spat_ctx_cost[i, j] = spatial_context_distance(
                    face, ps, faces, self.all_faces_by_image)

        # Normalize spatial context to [0, 1] range
        if spat_ctx_cost.max() > 0:
            spat_ctx_cost = spat_ctx_cost / (spat_ctx_cost.max() + 1e-8)

        # Learn weights from anchors
        alpha, beta = learn_weights_from_anchors(
            faces, available_sets, faces, self.all_faces_by_image, image_idx)

        # Combined cost
        if spat_map_cost is not None:
            # Normalize spatial map cost
            smc = spat_map_cost.copy()
            if smc.max() > 0:
                smc = smc / (smc.max() + 1e-8)
            combined = (self.w_emb * emb_cost +
                        self.w_spat * spat_ctx_cost +
                        self.w_hung * smc)
        else:
            # No spatial map available — redistribute weight to embedding
            combined = (self.w_emb + self.w_hung) * emb_cost + self.w_spat * spat_ctx_cost

        # Method D: Hungarian assignment
        matches = hungarian_assignment(combined, max_cost=self.embedding_threshold)

        matched_faces = set()
        matched_sets = set()

        # Apply matches — but verify with embedding distance
        for face_idx, set_idx in matches:
            if emb_cost[face_idx, set_idx] < self.embedding_threshold:
                available_sets[set_idx].add_face(faces[face_idx])
                matched_faces.add(face_idx)
                matched_sets.add(set_idx)

        # Also match recognized faces directly by SVM label (even if not Hungarian-matched)
        for i, face in enumerate(faces):
            if i in matched_faces:
                continue
            if face.svm_label == "Unknown":
                continue
            # Check if any available set already has this label
            for j, ps in enumerate(available_sets):
                if j in matched_sets:
                    continue
                if ps.label == face.svm_label and emb_cost[i, j] < self.embedding_threshold * 1.5:
                    ps.add_face(face)
                    matched_faces.add(i)
                    matched_sets.add(j)
                    break

        # Unmatched faces → new PersonSets
        for i, face in enumerate(faces):
            if i not in matched_faces:
                ps = PersonSet(faces=[face], person_id=self._new_person_id())
                if face.svm_label != "Unknown":
                    ps.label = face.svm_label
                self.person_sets.append(ps)

    def _merge_similar_sets(self):
        """Merge PersonSets with similar embeddings and no image overlap."""
        if len(self.person_sets) < 2:
            return

        merged = True
        while merged:
            merged = False
            n = len(self.person_sets)
            merge_pairs = []

            for i in range(n):
                for j in range(i + 1, n):
                    ps_i = self.person_sets[i]
                    ps_j = self.person_sets[j]

                    # Skip if they share any image
                    if ps_i.images_covered & ps_j.images_covered:
                        continue

                    # Skip if they have conflicting labels
                    if (ps_i.label and ps_j.label and
                            ps_i.label != ps_j.label):
                        continue

                    dist = cosine_distance(ps_i.mean_embedding, ps_j.mean_embedding)
                    if dist < self.merge_threshold:
                        merge_pairs.append((i, j, dist))

            if not merge_pairs:
                break

            # Merge closest pair
            merge_pairs.sort(key=lambda x: x[2])
            i, j, _ = merge_pairs[0]
            ps_i = self.person_sets[i]
            ps_j = self.person_sets[j]

            # Merge j into i
            for face in ps_j.faces:
                ps_i.add_face(face)
            if ps_j.label and not ps_i.label:
                ps_i.label = ps_j.label

            self.person_sets.pop(j)
            merged = True

    def _enforce_one_per_image(self):
        """
        If a PersonSet has multiple faces from the same image,
        keep only the best match (lowest embedding distance to set mean).
        Evicted faces get their own new PersonSets.
        """
        evicted = []
        for ps in self.person_sets:
            # Group faces by image
            by_img: Dict[int, List[FaceDetection]] = {}
            for f in ps.faces:
                by_img.setdefault(f.image_idx, []).append(f)

            keep_faces = []
            for img_idx, img_faces in by_img.items():
                if len(img_faces) == 1:
                    keep_faces.append(img_faces[0])
                else:
                    # Keep closest to mean embedding
                    mean_emb = ps.mean_embedding
                    best = min(img_faces,
                               key=lambda f: cosine_distance(f.embedding, mean_emb))
                    keep_faces.append(best)
                    evicted.extend([f for f in img_faces if f is not best])

            ps.faces = keep_faces

        # Create new sets for evicted faces
        for face in evicted:
            ps = PersonSet(faces=[face], person_id=self._new_person_id())
            if face.svm_label != "Unknown":
                ps.label = face.svm_label
            self.person_sets.append(ps)

    def _propagate_labels(self):
        """
        Label propagation: each PersonSet gets the mode of its members' SVM labels.
        Unlabeled sets get "Unknown Person #K".
        """
        unknown_counter = 1
        for ps in self.person_sets:
            labels = ps.svm_labels
            if labels:
                # Mode label
                from collections import Counter
                label_counts = Counter(labels)
                ps.label = label_counts.most_common(1)[0][0]
            else:
                ps.label = f"Unknown Person #{unknown_counter}"
                unknown_counter += 1

    def get_results(self) -> Dict:
        """Return structured results for all person sets."""
        results = {
            "person_sets": [],
            "stats": {
                "total_person_sets": len(self.person_sets),
                "labeled": sum(1 for ps in self.person_sets
                               if not ps.label.startswith("Unknown Person")),
                "unlabeled": sum(1 for ps in self.person_sets
                                 if ps.label.startswith("Unknown Person")),
                "total_faces": sum(len(ps.faces) for ps in self.person_sets),
                "embedding_threshold": float(self.embedding_threshold or 0),
            }
        }

        for ps in self.person_sets:
            ps_data = {
                "person_id": ps.person_id,
                "label": ps.label,
                "n_faces": len(ps.faces),
                "images": sorted(ps.images_covered),
                "faces": [
                    {
                        "image_file": f.image_file,
                        "image_idx": f.image_idx,
                        "bbox": [float(x) for x in f.bbox],
                        "svm_label": f.svm_label,
                        "svm_confidence": float(f.svm_confidence),
                        "det_score": float(f.det_score),
                    }
                    for f in ps.faces
                ]
            }
            results["person_sets"].append(ps_data)

        return results

    def get_attendance(self) -> Dict[str, dict]:
        """
        Build attendance dict from person sets.
        Returns {student_label: {status, images_detected_in, max_confidence, person_id}}
        """
        attendance = {}
        for ps in self.person_sets:
            if ps.label.startswith("Unknown Person"):
                continue
            max_conf = max(f.svm_confidence for f in ps.faces)
            imgs = sorted(ps.images_covered)
            attendance[ps.label] = {
                "status": "Present",
                "images_detected_in": len(imgs),
                "image_list": [f.image_file for f in ps.faces],
                "max_confidence": float(max_conf),
                "person_id": ps.person_id,
                "n_faces": len(ps.faces),
            }
        return attendance


# ── Convenience: Embedding-only Re-ID (ablation baseline) ────────────────────

class EmbeddingOnlyReIdentifier:
    """Simplified re-ID using only embedding clustering (no spatial methods)."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.person_sets: List[PersonSet] = []

    def process_all(self, detections_by_image: Dict[str, list],
                    predict_fn, image_files: List[str] = None) -> List[PersonSet]:
        if image_files is None:
            image_files = sorted(detections_by_image.keys())

        all_faces = []
        for idx, img_file in enumerate(image_files):
            for fi, det in enumerate(detections_by_image[img_file]):
                label, conf = predict_fn(det["embedding"])
                fd = FaceDetection(
                    image_idx=idx, image_file=img_file,
                    bbox=det["bbox"], embedding=det["embedding"],
                    det_score=det["det_score"],
                    kps=np.array(det["kps"]) if det.get("kps") else None,
                    svm_label=label, svm_confidence=conf, face_idx=fi,
                )
                all_faces.append(fd)

        if not all_faces:
            return []

        # Agglomerative clustering on embeddings
        embs = np.array([f.embedding for f in all_faces])
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        dist_matrix = cdist(embs_norm, embs_norm, metric="cosine")

        condensed = dist_matrix[np.triu_indices(len(all_faces), k=1)]
        Z = linkage(condensed, method="average")
        clusters = fcluster(Z, t=self.threshold, criterion="distance")

        # Build PersonSets from clusters
        cluster_map: Dict[int, List[FaceDetection]] = {}
        for face, cid in zip(all_faces, clusters):
            cluster_map.setdefault(cid, []).append(face)

        pid = 1
        unknown_count = 1
        for cid, faces in cluster_map.items():
            ps = PersonSet(faces=faces, person_id=pid)
            pid += 1

            # Label from mode of SVM labels
            labels = [f.svm_label for f in faces if f.svm_label != "Unknown"]
            if labels:
                from collections import Counter
                ps.label = Counter(labels).most_common(1)[0][0]
            else:
                ps.label = f"Unknown Person #{unknown_count}"
                unknown_count += 1
            self.person_sets.append(ps)

        return self.person_sets

    def get_attendance(self) -> Dict[str, dict]:
        attendance = {}
        for ps in self.person_sets:
            if ps.label.startswith("Unknown Person"):
                continue
            max_conf = max(f.svm_confidence for f in ps.faces)
            attendance[ps.label] = {
                "status": "Present",
                "images_detected_in": len(ps.images_covered),
                "image_list": [f.image_file for f in ps.faces],
                "max_confidence": float(max_conf),
                "person_id": ps.person_id,
                "n_faces": len(ps.faces),
            }
        return attendance
