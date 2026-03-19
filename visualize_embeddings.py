"""
visualize_embeddings.py - Interactive 3D visualization of face embeddings

Reduces 512-d embeddings to 3D using t-SNE, PCA, or UMAP, then generates
interactive plotly HTML files and rotating GIF animations.

Usage:
    python visualize_embeddings.py                          # uses face_database.pkl
    python visualize_embeddings.py --model face_database.pkl --method all
    python visualize_embeddings.py --pkl assets/embeddings_insightface.pkl
"""

import argparse
import os
import colorsys
import numpy as np
import pickle
from sklearn.manifold import TSNE, trustworthiness
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def load_from_model(model_path):
    """Load embeddings from face_database.pkl (consolidated format)."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    embs = data.get("embeddings")
    labels = data.get("embedding_labels")
    if embs is None or labels is None:
        raise ValueError(f"{model_path} has no embeddings. Retrain with updated face_model.py.")
    return embs, labels


def load_from_pkl(pkl_path):
    """Load embeddings from a standalone pkl file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["embeddings"], data["labels"]


def reduce_to_3d(embeddings, method="tsne", perplexity=30):
    """Reduce embeddings to 3D."""
    if method == "tsne":
        return TSNE(n_components=3, perplexity=min(perplexity, len(embeddings)-1),
                    random_state=42, max_iter=1000).fit_transform(embeddings)
    elif method == "pca":
        return PCA(n_components=3, random_state=42).fit_transform(embeddings)
    elif method == "umap":
        import umap
        return umap.UMAP(n_components=3, random_state=42).fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_metrics(embeddings_hd, coords_3d, labels):
    """Compute quality metrics for a dimensionality reduction."""
    le = LabelEncoder()
    y = le.fit_transform(labels)

    sil = silhouette_score(coords_3d, y)
    trust = trustworthiness(embeddings_hd, coords_3d, n_neighbors=5)

    # LOO k-NN accuracy in 3D space
    correct = 0
    for i in range(len(coords_3d)):
        X_train = np.delete(coords_3d, i, axis=0)
        y_train = np.delete(y, i)
        knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
        knn.fit(X_train, y_train)
        if knn.predict(coords_3d[i:i+1])[0] == y[i]:
            correct += 1
    knn_acc = correct / len(coords_3d)

    return {"silhouette": sil, "trustworthiness": trust, "knn_accuracy": knn_acc}


def generate_colors(n):
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        h = i / n
        s = 0.7 + 0.3 * (i % 2)
        v = 0.8 + 0.2 * ((i // 2) % 2)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append(f'rgb({int(r*255)},{int(g*255)},{int(b*255)})')
    return colors


def make_figure(coords_3d, labels, title):
    """Build a plotly 3D scatter figure."""
    import plotly.graph_objects as go

    unique_labels = sorted(set(labels))
    colors = generate_colors(len(unique_labels))
    label_to_color = {l: colors[i] for i, l in enumerate(unique_labels)}

    fig = go.Figure()
    for label in unique_labels:
        mask = np.array([l == label for l in labels])
        pts = coords_3d[mask]
        short = label.split()[0]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            name=short,
            text=[label] * len(pts),
            hovertemplate='%{text}<extra></extra>',
            marker=dict(size=4, color=label_to_color[label], opacity=0.85),
        ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='Dim 1', yaxis_title='Dim 2', zaxis_title='Dim 3'),
        width=1000, height=750,
        legend=dict(font=dict(size=9), itemsizing='constant'),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_interactive_3d(coords_3d, labels, title, output_path):
    """Create an interactive 3D scatter plot with plotly."""
    fig = make_figure(coords_3d, labels, title)
    fig.write_html(output_path)
    print(f"Saved: {output_path}")


def generate_rotating_gif(coords_3d, labels, title, output_path, n_frames=36):
    """Generate a rotating GIF of the 3D scatter plot."""
    fig = make_figure(coords_3d, labels, title)
    fig.update_layout(showlegend=False, width=600, height=500,
                      margin=dict(l=0, r=0, t=30, b=0))

    frames = []
    for i in range(n_frames):
        angle = i * (360 / n_frames)
        fig.update_layout(scene_camera=dict(
            eye=dict(
                x=1.5 * np.cos(np.radians(angle)),
                y=1.5 * np.sin(np.radians(angle)),
                z=0.6,
            )
        ))
        img_bytes = fig.to_image(format="png", scale=1)
        from PIL import Image
        import io
        frames.append(Image.open(io.BytesIO(img_bytes)))

    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=100, loop=0)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive 3D visualization of face embeddings")
    parser.add_argument("--model", default="face_database.pkl",
                        help="Path to face_database.pkl (default)")
    parser.add_argument("--pkl", default=None,
                        help="Path to standalone embeddings .pkl file (alternative)")
    parser.add_argument("--method", choices=["tsne", "pca", "umap", "all"], default="all",
                        help="Dimensionality reduction method")
    parser.add_argument("--output-dir", default="assets", help="Output directory")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation")
    parser.add_argument("--no-metrics", action="store_true", help="Skip metrics computation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    methods = ["tsne", "pca", "umap"] if args.method == "all" else [args.method]

    if args.pkl:
        embs, labels = load_from_pkl(args.pkl)
        name = os.path.splitext(os.path.basename(args.pkl))[0]
    else:
        embs, labels = load_from_model(args.model)
        name = "insightface"

    print(f"Loaded {len(embs)} embeddings of dim {embs.shape[1]}")

    all_metrics = {}

    for method in methods:
        print(f"\n  {method}...", flush=True)
        try:
            coords = reduce_to_3d(embs, method)

            # Interactive HTML
            plot_interactive_3d(
                coords, labels,
                title=f"InsightFace Embeddings ({method.upper()} 3D)",
                output_path=os.path.join(args.output_dir, f"embeddings_3d_{name}_{method}.html"),
            )

            # Rotating GIF
            if not args.no_gif:
                generate_rotating_gif(
                    coords, labels,
                    title=f"{method.upper()} 3D",
                    output_path=os.path.join(args.output_dir, f"embeddings_3d_{name}_{method}.gif"),
                )

            # Metrics
            if not args.no_metrics:
                m = compute_metrics(embs, coords, labels)
                all_metrics[method.upper()] = m
                print(f"    Silhouette: {m['silhouette']:.4f}  "
                      f"Trustworthiness: {m['trustworthiness']:.4f}  "
                      f"3D k-NN: {m['knn_accuracy']:.1%}")

        except Exception as e:
            print(f"  failed: {e}")

    if all_metrics:
        print(f"\n{'Method':<8} {'Silhouette':>11} {'Trustworth':>12} {'3D k-NN Acc':>12}")
        print("-" * 46)
        for method, m in all_metrics.items():
            print(f"{method:<8} {m['silhouette']:>11.4f} {m['trustworthiness']:>12.4f} {m['knn_accuracy']:>11.1%}")
