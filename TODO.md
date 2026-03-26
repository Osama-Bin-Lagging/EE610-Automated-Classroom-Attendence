# Future Research Directions (Accuracy-Focused)

## Result Improvement Methods
Directly improve recall/precision on the current pipeline without major architectural changes.

### 1. Data Augmentation Impact Study
`augment.py` exists but is untested. Benchmark LOO accuracy and classroom recall with augmented training data — geometric (flips, rotation +/-15 deg, random crops) and photometric (brightness, contrast, Gaussian blur, noise). Key question: does expanding from 5 to 25 images per student via augmentation improve the 95.7% classroom recall?

### 2. Failure Analysis & Targeted Fix for the 2 Missed Students
The top strategies consistently miss 2/47. Profile these failures — are they occluded, extreme side-profile, low-resolution, or seated in detector blind spots? Annotate per-image per-student detection success/failure matrix across all 12 validation images.

### 3. Test-Time Augmentation (TTA) for Recognition
At inference, create multiple augmented versions of each detected face (horizontal flip, slight rotation, brightness shifts), extract embeddings for each, and average or max-pool the embeddings before SVM classification. Stabilizes predictions on hard/noisy detections at no training cost.

### 4. Ensemble of Multiple Recognition Models
InsightFace and FaceNet both achieve ~100% LOO. Ensemble their embeddings — concatenate (512+512=1024-d) or average their similarity scores — for more robust classroom predictions. The two models have different failure modes, so ensembling should strictly improve recall.

### 5. Multi-Frame Temporal Aggregation
Currently aggregating across 12 images (present if identified in any single image). Improve this: try embedding averaging across frames, confidence-weighted voting, or requiring k-of-n agreement. Quantify: how many images are needed for 100% recall?

### 6. Face Quality-Aware Filtering & Re-ranking
Score each detected face on quality (detection confidence, face area, estimated pose angle, blur score) and weight recognition confidence accordingly. Low-quality detections with borderline SVM scores get suppressed (improving precision) while high-quality ones get boosted. Look at SER-FIQ for unsupervised quality estimation from embedding perturbation stability.

### 7. Cross-Pose & Multi-View Enrollment
Current enrollment uses 5 frontal-ish photos. If the missed students are always at extreme angles, adding 2-3 side-profile enrollment images per student could directly fix those failures. Evaluate: how much does adding 1 profile-angle enrollment image per student improve classroom recall?

### 8. Hard Negative Mining in Embedding Space
Analyze which student pairs are closest in embedding space (most confusable). Generate targeted augmentations or collect additional enrollment data specifically for these hard pairs. Train the SVM with emphasis on separating the most confusable identities.

### 9. Self-Training / Pseudo-Labeling from Classroom Images
Use current high-confidence predictions on classroom images as pseudo-labels. Add these classroom crops (real conditions, real angles) to the training set and retrain the SVM. Iteratively: predict, add high-confidence detections to training, retrain, predict again. Bridges the enrollment-to-classroom domain gap without manual labeling.

## R&D Work
Deeper research directions requiring new models, architectures, or significant pipeline changes.

### 10. Super-Resolution on Small/Distant Faces
Many missed classroom faces are small (sub-50px) from students in back rows. Apply face super-resolution (Real-ESRGAN, GFPGAN, or CodeFormer) on detected crops before embedding extraction. Hypothesis: upscaling small face crops from 30x30 to 112x112 with a face-aware SR model should produce better embeddings than naive bilinear resize.

### 11. Metric Learning Fine-Tuning on Your Domain
ArcFace is pretrained on MS1MV2 (celebrity faces, controlled conditions). Fine-tune the last 1-2 layers on the 290-image dataset using ArcFace/CosFace loss with a small learning rate (1e-5). The domain gap between enrollment photos (controlled, frontal) and classroom photos (angled, distant, varied lighting) is the main accuracy bottleneck.

### 12. Open-Set Recognition with Embedding Distance
Replace SVM probability thresholds (poorly calibrated at 2-4% across 58 classes) with cosine distance to nearest class centroid in embedding space. Set a distance threshold (e.g., 0.4 for ArcFace) to reject unknowns. Compare with Extreme Value Theory (OpenMax) for principled open-set rejection.

### 13. Body/Context-Based Re-identification as Fallback
For faces that fail detection or recognition, use upper-body or seating-position features as a secondary signal. Person re-identification models (e.g., OSNet, TransReID) trained on body crops can complement face recognition for students whose faces are occluded or turned away.

### 14. Pose-Adaptive Detection Pipeline
Estimate head pose angle for each detection and route to pose-specific processing — frontal faces go through standard ArcFace, profile faces get pose-normalized (3DDFA-based frontalization or pose-specific embedding models) before recognition. Specifically targets missed students if they are profile failures.

### 15. Low-Quality / Degraded Face Recognition Models
Swap in recognition models specifically trained on low-resolution, surveillance-style, or degraded imagery (e.g., AdaFace, QualFace, or models trained on TinyFace/SCface datasets). Standard ArcFace is trained on high-quality web photos — models designed for low-quality inputs may extract more discriminative embeddings from small, blurry, or poorly-lit classroom crops without needing super-resolution preprocessing.
