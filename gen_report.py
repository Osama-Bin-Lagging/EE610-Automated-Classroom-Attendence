"""Generate progress report PDF using fpdf2."""
from fpdf import FPDF

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=10)
pdf.set_margins(18, 12, 18)
pdf.add_page()

LM = pdf.l_margin

# Title
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "EE610 - Automated Classroom Attendance System", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 11)
pdf.cell(0, 7, "Progress Report", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)

# Team
pdf.set_font("Helvetica", "", 9.5)
pdf.cell(0, 5, "Manit Jhajharia (23B1265)   |   Aboli G. Malshikare (23B1211)   |   Shreya Nigam (23B1258)", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, "Chhavi Yadav (23B3923)   |   Yashasvee V. Taiwade (23B2232)", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)

pdf.set_draw_color(160, 160, 160)
pdf.line(LM, pdf.get_y(), 210 - LM, pdf.get_y())
pdf.ln(5)

def section(title):
    pdf.set_x(LM)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

def subsection(title):
    pdf.set_x(LM)
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

def body(text):
    pdf.set_x(LM)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, text)
    pdf.ln(2)

# Section 1
section("1. Work Completed")

body(
    "We are building an automated attendance system that processes classroom photographs to identify "
    "students using face recognition. Our dataset consists of 58 students with 5 enrollment images "
    "each (290 total). We validate on 12 real 4K classroom photographs with ~49 students visible."
)

subsection("Recognition Benchmarking")
body(
    "We benchmarked 10 recognition approaches across three tiers using leave-one-out cross-validation "
    "with SVM and KNN classifiers: deep face embeddings (InsightFace, FaceNet, DeepFace variants), "
    "generic ImageNet features (EfficientNet-B0, ResNet-50), and classical CV (HOG, Eigenfaces, "
    "LBPH, Fisherfaces). InsightFace achieved 100% LOO accuracy. Generic features reached 89%. "
    "Classical methods plateaued at ~55%."
)

subsection("Detection Benchmarking")
body(
    "The bottleneck is face localization -- the default detector misses ~40% of faces in 4K classroom "
    "images. We benchmarked 10 detection strategies including RetinaFace variants, Haar cascades, "
    "MTCNN, and hybrid approaches. Five strategies tied at 95.7% recall (45/47 students). "
    "Haar Aggressive was fastest at 3.9s/image, 10x faster than tiled RetinaFace."
)

subsection("Preprocessing, Visualization & Web UI")
body(
    "An automated pipeline detects, aligns, and crops faces to 112x112 for training, with data "
    "augmentation for robustness. 512-d embeddings were visualized in 3D using t-SNE, PCA, and "
    "UMAP. A Streamlit web app provides the end-to-end attendance workflow."
)

pdf.set_draw_color(160, 160, 160)
pdf.line(LM, pdf.get_y(), 210 - LM, pdf.get_y())
pdf.ln(4)

# Section 2
section("2. Individual Contributions")

contributions = [
    ("Manit Jhajharia",
     "Designed the core recognition pipeline -- SVM classifier on ArcFace embeddings with tuned "
     "thresholds. Benchmarked InsightFace and facenet-pytorch. Developed union and cascade "
     "detection strategies. Set up validation framework with ground truth matching."),
    ("Aboli Ganesh Malshikare",
     "Built the preprocessing pipeline -- face detection, alignment, cropping, and normalization. "
     "Evaluated DeepFace ArcFace and GhostFaceNet. Benchmarked Haar cascade detectors at various "
     "sensitivity levels for classroom localization."),
    ("Shreya Nigam",
     "Developed the Streamlit web interface for the attendance workflow. Investigated transfer "
     "learning from ImageNet-pretrained models (EfficientNet-B0, ResNet-50). Benchmarked MTCNN "
     "detection."),
    ("Chhavi Yadav",
     "Created 3D embedding visualization using t-SNE, PCA, and UMAP with quality metrics. "
     "Evaluated HOG descriptors and Eigenfaces. Established RetinaFace detection baselines."),
    ("Yashasvee Vijay Taiwade",
     "Implemented data augmentation with geometric and photometric transforms. Built the LBPH "
     "baseline and evaluated Fisherfaces. Benchmarked tiled RetinaFace detection."),
]

for name, desc in contributions:
    pdf.set_x(LM)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, name, new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(LM)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 4.5, desc)
    pdf.ln(1)

pdf.set_draw_color(160, 160, 160)
pdf.line(LM, pdf.get_y(), 210 - LM, pdf.get_y())
pdf.ln(2)
pdf.set_font("Helvetica", "I", 9)
pdf.set_x(LM)
pdf.cell(0, 4, "Repository: github.com/Osama-Bin-Lagging/EE610-Automated-Classroom-Attendence", align="C")

pdf.output("report.pdf")
print("report.pdf generated")
