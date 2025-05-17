import psutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd
import os
import time
from store_faces import store_face

TEST_IMAGE_DIR = "test_images"
IMAGE_DIR = "images_with_filters"
INDEX_FILE = "fais_db/filtered_face_index.faiss"
METADATA_FILE = "fais_db/metadata.txt"
OUTPUT_DIR = "benchmarks_store"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def benchmark_store_face():
    search_times = []
    memory_usages = []
    cpu_usages = []
    filenames = []

    for filename in os.listdir(TEST_IMAGE_DIR):
        if filename.endswith(".png"):
            image_path = os.path.join(TEST_IMAGE_DIR, filename)
            process = psutil.Process(os.getpid())
            start_time = time.time()
            cpu_start = psutil.cpu_percent(interval=None)
            store_face(image_path)
            cpu_end = psutil.cpu_percent(interval=None)
            end_time = time.time()
            elapsed_time = end_time - start_time
            mem = process.memory_info().rss / (1024 ** 2)
            search_times.append(elapsed_time)
            memory_usages.append(mem)
            cpu_usages.append(cpu_end - cpu_start)
            filenames.append(filename)

    # Save data to CSV
    df = pd.DataFrame({
        "Image": filenames,
        "Store Time (s)": search_times,
        "Memory Usage (MB)": memory_usages,
        "CPU Usage (%)": cpu_usages
    })
    df.to_csv(os.path.join(OUTPUT_DIR, "store_face_metrics.csv"), index=False)

    # Plotting - Bar Chart
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(search_times)), search_times, tick_label=filenames)
    plt.xticks(rotation=45)
    plt.ylabel("Store Time (s)")
    plt.title("Store Time per Image")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "store_time_per_image.png"))
    plt.close()

    # Plotting - Boxplot
    plt.figure(figsize=(8, 4))
    plt.boxplot(search_times, vert=False)
    plt.title("Distribution of Store Times")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "store_time_boxplot.png"))
    plt.close()

    # Plotting - Line Chart of Store Times Over Sequence
    plt.figure(figsize=(10, 5))
    plt.plot(search_times, marker='o', linestyle='-')
    plt.title("Store Time Over Sequence")
    plt.xlabel("Image Index")
    plt.ylabel("Store Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "store_time_lineplot.png"))
    plt.close()

    # 3D Plot
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
        text=filenames,
        hoverinfo='text'
    )
    layout = go.Layout(
        title="3D Benchmark: Store Time vs Memory vs CPU",
        scene=dict(
            xaxis_title='Store Time (s)',
            yaxis_title='Memory Usage (MB)',
            zaxis_title='CPU Usage (%)',
            bgcolor='rgb(10,10,10)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig = go.Figure(data=[trace], layout=layout)
    fig.write_html(os.path.join(OUTPUT_DIR, "store_face_3D.html"))
    fig.write_image(os.path.join(OUTPUT_DIR, "store_face_3D.png"))

benchmark_store_face()
