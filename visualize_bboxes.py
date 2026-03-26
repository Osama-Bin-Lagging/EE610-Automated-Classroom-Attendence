"""
visualize_bboxes.py - Draw detected face bounding boxes on validation images.

For each val image, draws every detected bbox with:
- Green box + predicted name if recognized
- Red box + "Unknown (conf%)" if below threshold
Saves annotated images to results/bbox_overlay/
"""

import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import numpy as np
from face_model import FaceRecognitionModel, load_image_rgb
from cache_detections import load_cache
from benchmark_detection import VAL_DIR

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_DIR, "results", "bbox_overlay")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    enrollment_embs, val_detections, image_files = load_cache()

    # Train classifier
    model = FaceRecognitionModel(threshold=0.05)
    model.train(embeddings_dict=enrollment_embs)

    for img_file in image_files:
        rgb = load_image_rgb(os.path.join(VAL_DIR, img_file))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        faces = val_detections[img_file]
        n_recognized = 0

        for face in faces:
            x1, y1, x2, y2 = [int(c) for c in face["bbox"]]
            emb = face["embedding"]
            det_score = face.get("det_score", 0)

            label, conf = model.predict(emb)

            if label != "Unknown":
                color = (0, 200, 0)  # green
                text = f"{label} ({conf:.1%})"
                n_recognized += 1
            else:
                color = (0, 0, 220)  # red
                text = f"? ({conf:.1%} d={det_score:.2f})"

            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)

            # Text background
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.4
            thickness = 1
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            cv2.rectangle(bgr, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(bgr, text, (x1, y1 - 2), font, scale, (255, 255, 255), thickness)

        out_path = os.path.join(OUT_DIR, img_file)
        cv2.imwrite(out_path, bgr)
        print(f"  {img_file}: {len(faces)} boxes ({n_recognized} recognized)")

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
