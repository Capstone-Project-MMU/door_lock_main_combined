import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import warnings
import time
import psutil
import pandas as pd

from search_faces import search_face, get_embedding

# Create output directory
output_dir = "benchmarks_3d"
os.makedirs(output_dir, exist_ok=True)

def generate_visual_benchmark_from_search(directory="test_images", k=5, runs=5, method="tsne"):
    embeddings = []
    labels = []

    query_images = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(".png")]

    for query_image in query_images:
        for run in range(runs):
            result = search_face(query_image, k=k)
            matches = result.get("matches", [])
            for match in matches:
                emb = get_embedding(match["image_path"])
                if emb is not None:
                    embeddings.append(emb)
                    labels.append(f"{os.path.basename(query_image)} | Run {run+1} | Rank {match['rank']}")

    if len(embeddings) < 5:
        print("Not enough embeddings found to visualize.")
        return

    X = np.vstack(embeddings)
    reducer_2d = PCA(n_components=2) if method == 'pca' else TSNE(n_components=2, perplexity=30, random_state=42)
    reducer_3d = PCA(n_components=3) if method == 'pca' else TSNE(n_components=3, perplexity=30, random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_2d = reducer_2d.fit_transform(X)
        X_3d = reducer_3d.fit_transform(X)

    # --- 2D Plot ---
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 7))
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=range(len(X_2d)),
        cmap='plasma',
        s=50,
        alpha=0.85,
        edgecolor='white'
    )
    plt.colorbar(scatter, label='Point Index')
    plt.title(f"2D {method.upper()} Projection of Face Matches", fontsize=14)
    plt.xlabel("Face Feature X")
    plt.ylabel("Face Feature Y")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "search_face_2D.png"))
    plt.close()

    # --- 3D Plot with Plotly ---
    trace = go.Scatter3d(
        x=X_3d[:, 0],
        y=X_3d[:, 1],
        z=X_3d[:, 2],
        mode='markers',
        text=labels,
        marker=dict(
            size=8,
            color=np.linspace(0, 1, len(X_3d)),
            colorscale='Plasma',
            colorbar=dict(title='Point Index'),
            opacity=0.9,
            line=dict(width=1, color='black')
        ),
        hoverinfo='text'
    )
    layout = go.Layout(
        title=f"3D {method.upper()} Projection of Face Matches",
        scene=dict(
            xaxis_title='Face Feature X',
            yaxis_title='Face Feature Y',
            zaxis_title='Face Feature Z',
            bgcolor='rgb(10,10,10)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig = go.Figure(data=[trace], layout=layout)
    fig.write_html(os.path.join(output_dir, "search_face_3D.html"))
    fig.write_image(os.path.join(output_dir, "search_face_3D.png"))

def insights():
    query_images_dir = "test_images"
    query_images = [os.path.join(query_images_dir, f) for f in os.listdir(query_images_dir) if f.lower().endswith('.png')]

    search_times = []
    memory_usages = []
    cpu_usages = []

    for image in query_images:
        process = psutil.Process(os.getpid())
        start_time = time.time()
        cpu_start = psutil.cpu_percent(interval=None)
        search_face(image, k=5)
        cpu_end = psutil.cpu_percent(interval=None)
        end_time = time.time()
        elapsed_time = end_time - start_time
        mem = process.memory_info().rss / (1024 ** 2)
        search_times.append(elapsed_time)
        memory_usages.append(mem)
        cpu_usages.append(cpu_end - cpu_start)

    # Save metrics to CSV
    df = pd.DataFrame({
        "Query Image": [os.path.basename(i) for i in query_images],
        "Search Time (s)": search_times,
        "Memory Usage (MB)": memory_usages,
        "CPU Usage (%)": cpu_usages,
    })
    df.to_csv(os.path.join(output_dir, "search_metrics.csv"), index=False)

    # --- Benchmark 2: Time vs K ---
    k_values = [1, 5, 10, 20, 50]
    avg_times_k = []

    for k in k_values:
        times = []
        for image in query_images[:3]:
            start = time.time()
            search_face(image, k=k)
            end = time.time()
            times.append(end - start)
        avg_times_k.append(np.mean(times))

    # --- Benchmark 3: Cumulative Runtime ---
    cumulative_times = []
    runs = 10
    total_time = 0
    for i in range(runs):
        start = time.time()
        search_face(query_images[i % len(query_images)], k=5)
        end = time.time()
        total_time += (end - start)
        cumulative_times.append(total_time)

    # Save plots
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(range(len(search_times)), search_times, tick_label=[os.path.basename(i) for i in query_images])
    ax1.set_ylabel("Search Time (s)")
    ax1.set_xlabel("Query Image")
    ax1.set_title("Search Time per Image")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, "benchmark_time_per_image.png"))
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(k_values, avg_times_k, marker='o')
    ax2.set_xlabel("Top K Matches")
    ax2.set_ylabel("Avg Search Time (s)")
    ax2.set_title("Search Time vs. K")
    ax2.grid(True)
    fig2.savefig(os.path.join(output_dir, "benchmark_time_vs_k.png"))
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(range(1, runs+1), cumulative_times, marker='x', linestyle='--')
    ax3.set_xlabel("Run")
    ax3.set_ylabel("Cumulative Time (s)")
    ax3.set_title("Cumulative Runtime Over Runs")
    ax3.grid(True)
    fig3.savefig(os.path.join(output_dir, "benchmark_cumulative_runtime.png"))
    plt.close(fig3)

    # --- 3D Plot for Insights ---
    colors = np.linspace(0, 1, len(search_times))
    trace = go.Scatter3d(
        x=search_times,
        y=memory_usages,
        z=cpu_usages,
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            colorscale='Plasma',
            colorbar=dict(title='Relative Index'),
            opacity=0.9
        ),
        text=[f"Index {i}" for i in range(len(search_times))],
        hoverinfo='text'
    )
    layout = go.Layout(
        title="3D Benchmark: Time vs Memory vs CPU",
        scene=dict(
            xaxis_title='Search Time (s)',
            yaxis_title='Memory Usage (MB)',
            zaxis_title='CPU Usage (%)',
            bgcolor='rgb(10,10,10)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig = go.Figure(data=[trace], layout=layout)
    fig.write_html(os.path.join(output_dir, "benchmark_3D_metrics.html"))
    fig.write_image(os.path.join(output_dir, "benchmark_3D_metrics.png"))

if __name__ == "__main__":
    generate_visual_benchmark_from_search(directory="test_images", k=5, runs=5, method="tsne")
    insights()
