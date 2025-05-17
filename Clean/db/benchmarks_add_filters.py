import os
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd
from add_filters import apply_filters  

def benchmark_apply_filters(test_dir, output_dir, func_to_benchmark):
    filenames = []
    apply_times = []
    memory_usages = []
    cpu_usages = []

    for filename in os.listdir(test_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(test_dir, filename)
            person_name = os.path.splitext(filename)[0]
            process = psutil.Process(os.getpid())

            start_time = time.time()
            cpu_start = psutil.cpu_percent(interval=None)
            func_to_benchmark(image_path, output_dir, person_name)
            cpu_end = psutil.cpu_percent(interval=None)
            end_time = time.time()

            apply_times.append(end_time - start_time)
            memory_usages.append(process.memory_info().rss / (1024 ** 2))  # in MB
            cpu_usages.append(cpu_end - cpu_start)
            filenames.append(filename)

    # Save CSV
    df = pd.DataFrame({
        "Image": filenames,
        "Filter Apply Time (s)": apply_times,
        "Memory Usage (MB)": memory_usages,
        "CPU Usage (%)": cpu_usages
    })
    os.makedirs("benchmarks_filters", exist_ok=True)
    df.to_csv("benchmarks_filters/filter_metrics.csv", index=False)

    # Bar Plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(apply_times)), apply_times, tick_label=filenames)
    plt.xticks(rotation=45)
    plt.ylabel("Filter Apply Time (s)")
    plt.title("Filter Time per Image")
    plt.tight_layout()
    plt.savefig("benchmarks_filters/filter_time_bar.png")
    plt.close()

    # Boxplot
    plt.figure(figsize=(8, 4))
    plt.boxplot(apply_times, vert=False)
    plt.title("Distribution of Filter Times")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("benchmarks_filters/filter_time_boxplot.png")
    plt.close()

    # Line Plot
    plt.figure(figsize=(10, 5))
    plt.plot(apply_times, marker='o', linestyle='-')
    plt.title("Filter Apply Time Over Sequence")
    plt.xlabel("Image Index")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    plt.savefig("benchmarks_filters/filter_time_lineplot.png")
    plt.close()

    # 3D Plot
    colors = np.linspace(0, 1, len(apply_times))
    trace = go.Scatter3d(
        x=apply_times,
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
        title="3D Benchmark: Filter Time vs Memory vs CPU",
        scene=dict(
            xaxis_title='Apply Time (s)',
            yaxis_title='Memory Usage (MB)',
            zaxis_title='CPU Usage (%)',
            bgcolor='rgb(10,10,10)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig = go.Figure(data=[trace], layout=layout)
    fig.write_html("benchmarks_filters/filter_3D.html")
    fig.write_image("benchmarks_filters/filter_3D.png")

benchmark_apply_filters(
    test_dir="test_images",
    output_dir="images_with_filters",
    func_to_benchmark=apply_filters
)
