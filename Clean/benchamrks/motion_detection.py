import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.pyplot as plt


def calculate_difference(image1, image2):
    """
    Check if two images are different.
    
    Args:
        image1 (numpy.ndarray): The first image.
        image2 (numpy.ndarray): The second image.
    
    Returns:
        bool: True if there's any difference, False otherwise.
    """
    difference = np.abs(image1.astype(np.int16) - image2.astype(np.int16))
    return np.any(difference != 0)

def plot_image_comparison_3d(image_dir, output_dir="image_comparison_3d_output"):
    os.makedirs(output_dir, exist_ok=True)

    # Load and sort image files
    image_files = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if len(image_files) < 2:
        print("Not enough images to compare.")
        return

    image_vectors = []
    labels = []
    pair_indices = []
    valid_pairs = []

    pair_counter = 0

    for i in range(len(image_files) - 1):
        path1, path2 = image_files[i], image_files[i + 1]
        img1 = Image.open(path1).convert("L").resize((64, 64))
        img2 = Image.open(path2).convert("L").resize((64, 64))

        arr1 = np.array(img1)
        arr2 = np.array(img2)

        if calculate_difference(arr1, arr2):
            image_vectors.append(arr1.flatten())
            image_vectors.append(arr2.flatten())
            labels.append(os.path.basename(path1))
            labels.append(os.path.basename(path2))
            pair_indices.extend([pair_counter, pair_counter])
            valid_pairs.append((len(image_vectors) - 2, len(image_vectors) - 1))
            pair_counter += 1

    if len(image_vectors) == 0:
        print("No differing image pairs found.")
        return

    image_vectors = np.array(image_vectors)

    # PCA to 3D
    reducer = PCA(n_components=3)
    X_3d = reducer.fit_transform(image_vectors)

    # Color mapping
    unique_pairs = len(set(pair_indices))
    color_values = np.linspace(0, 1, unique_pairs)
    color_map = {pair: val for pair, val in zip(range(unique_pairs), color_values)}
    assigned_colors = [color_map[p] for p in pair_indices]

    # 3D points
    scatter = go.Scatter3d(
        x=X_3d[:, 0],
        y=X_3d[:, 1],
        z=X_3d[:, 2],
        mode='markers+text',
        text=labels,
        marker=dict(
            size=7,
            color=assigned_colors,
            colorscale='Rainbow',
            opacity=0.9,
            line=dict(width=1, color='black')
        ),
        hoverinfo='text'
    )

    # Lines between pairs
    lines = []
    for idx1, idx2 in valid_pairs:
        lines.append(go.Scatter3d(
            x=[X_3d[idx1, 0], X_3d[idx2, 0]],
            y=[X_3d[idx1, 1], X_3d[idx2, 1]],
            z=[X_3d[idx1, 2], X_3d[idx2, 2]],
            mode='lines',
            line=dict(color='white', width=2),
            hoverinfo='none',
            showlegend=False
        ))

    layout = go.Layout(
        title="3D Projection of Differing Image Pairs",
        scene=dict(
            xaxis_title='PCA X',
            yaxis_title='PCA Y',
            zaxis_title='PCA Z',
            bgcolor='rgb(10,10,10)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig = go.Figure(data=[scatter] + lines, layout=layout)
    fig.write_html(os.path.join(output_dir, "diff_image_pairs_3d.html"))
    fig.write_image(os.path.join(output_dir, "diff_image_pairs_3d.png"))
    
    save_additional_graphs(X_3d, valid_pairs, output_dir)
    print(f"Saved visualization to {output_dir}")

def save_additional_graphs(embeddings_3d, valid_pairs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Count of differing pairs
    total_diffs = len(valid_pairs)
    fig1, ax1 = plt.subplots()
    ax1.bar(["Differing Pairs"], [total_diffs], color='orange')
    ax1.set_title("Number of Differing Image Pairs")
    fig1.savefig(os.path.join(output_dir, "num_differing_pairs.png"))
    plt.close(fig1)

    # Euclidean distance histogram
    distances = [
        np.linalg.norm(embeddings_3d[i] - embeddings_3d[j])
        for i, j in valid_pairs
    ]
    fig2, ax2 = plt.subplots()
    ax2.hist(distances, bins=10, color='skyblue', edgecolor='black')
    ax2.set_title("Histogram of Pairwise Distances")
    ax2.set_xlabel("Euclidean Distance")
    ax2.set_ylabel("Frequency")
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "histogram_distances.png"))
    plt.close(fig2)

    # Sorted distances
    fig3, ax3 = plt.subplots()
    sorted_distances = sorted(distances, reverse=True)
    ax3.plot(sorted_distances, marker='o', linestyle='--', color='green')
    ax3.set_title("Sorted Pairwise Distances")
    ax3.set_xlabel("Pair Index")
    ax3.set_ylabel("Distance")
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "sorted_distances.png"))
    plt.close(fig3)

    # Boxplot
    fig4, ax4 = plt.subplots()
    ax4.boxplot(distances, patch_artist=True, boxprops=dict(facecolor='purple'))
    ax4.set_title("Boxplot of Pairwise Distances")
    ax4.set_ylabel("Distance")
    fig4.tight_layout()
    fig4.savefig(os.path.join(output_dir, "boxplot_distances.png"))
    plt.close(fig4)
    print(f"Saved additional graphs to {output_dir}")

# Example usage
if __name__ == "__main__":
    plot_image_comparison_3d("/Volumes/main/Capstone/Clean/db/images_with_filters")

