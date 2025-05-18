import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mediapipe as mp
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, classification_report
)
import plotly.graph_objs as go


def detect_faces(frame):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.4)
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = (
                int(bboxC.xmin * w),
                int(bboxC.ymin * h),
                int(bboxC.width * w),
                int(bboxC.height * h),
            )
            x, y, width, height = max(0, x), max(0, y), max(1, width), max(1, height)
            face = frame[y: y + height, x: x + width]
            faces.append((x, y, width, height, face))
    return faces


def evaluate_face_detector(image_dir, output_dir="face_detection_benchmark"):
    os.makedirs(output_dir, exist_ok=True)

    y_true = []
    y_pred = []
    inference_times = []
    image_names = []

    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(image_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue

        has_face_gt = bool(re.search(r"moh|mohamed", fname, re.IGNORECASE))
        y_true.append(1 if has_face_gt else 0)

        start_time = cv2.getTickCount()
        faces = detect_faces(img)
        end_time = cv2.getTickCount()
        time_taken = (end_time - start_time) / cv2.getTickFrequency()
        inference_times.append(time_taken)

        has_face_pred = len(faces) > 0
        y_pred.append(1 if has_face_pred else 0)

        image_names.append(fname)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Save CSV
    df = pd.DataFrame({
        "Filename": image_names,
        "Ground Truth": y_true,
        "Prediction": y_pred,
        "Inference Time (s)": inference_times
    })
    df.to_csv(os.path.join(output_dir, "face_detection_metrics.csv"), index=False)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Face", "Face"])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(os.path.join(output_dir, "precision_recall.png"))
    plt.close()

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=["No Face", "Face"], output_dict=True)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Classification Report")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classification_report.png"))
    plt.close()

    # Face vs Non-face GT distribution
    fig, ax = plt.subplots()
    ax.bar(["No Face (GT)", "Face (GT)"], [np.sum(y_true == 0), np.sum(y_true == 1)], color=["gray", "green"])
    ax.set_title("Ground Truth Face Distribution")
    fig.savefig(os.path.join(output_dir, "face_gt_distribution.png"))
    plt.close(fig)

    # Prediction Pie Chart
    fig, ax = plt.subplots()
    ax.pie([np.sum(y_pred == 0), np.sum(y_pred == 1)], labels=["No Face", "Face"], autopct='%1.1f%%', startangle=90)
    ax.set_title("Prediction Result Breakdown")
    fig.savefig(os.path.join(output_dir, "prediction_distribution.png"))
    plt.close(fig)

    # Inference time bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(image_names, inference_times, color='orange')
    ax.set_title("Inference Time Per Image")
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Image")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "inference_time_per_image.png"))
    plt.close()

    # Cumulative inference time
    cumulative_time = np.cumsum(inference_times)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumulative_time)+1), cumulative_time, marker='o', linestyle='--', color='blue')
    ax.set_title("Cumulative Inference Time")
    ax.set_xlabel("Image Index")
    ax.set_ylabel("Total Time (s)")
    fig.savefig(os.path.join(output_dir, "cumulative_inference_time.png"))
    plt.close()

    # 3D scatter: Index vs Time vs Prediction
    trace = go.Scatter3d(
        x=list(range(len(inference_times))),
        y=inference_times,
        z=y_pred,
        mode='markers',
        marker=dict(
            size=6,
            color=y_true,
            colorscale='Viridis',
            colorbar=dict(title='Ground Truth'),
            line=dict(width=1)
        ),
        text=image_names,
        hoverinfo='text'
    )
    layout = go.Layout(
        title="3D Visualization: Index vs Time vs Prediction",
        scene=dict(
            xaxis_title='Image Index',
            yaxis_title='Inference Time (s)',
            zaxis_title='Prediction (0=No Face, 1=Face)',
            bgcolor='rgb(10,10,10)'
        )
    )
    fig3d = go.Figure(data=[trace], layout=layout)
    fig3d.write_html(os.path.join(output_dir, "inference_3d_plot.html"))
    fig3d.write_image(os.path.join(output_dir, "inference_3d_plot.png"))

    # 3D ROC surface
    trace2 = go.Scatter3d(
        x=thresholds,
        y=fpr,
        z=tpr,
        mode='lines+markers',
        marker=dict(size=3, color=tpr, colorscale='Bluered', colorbar=dict(title='TPR')),
        line=dict(width=2)
    )
    layout2 = go.Layout(
        title="3D ROC Threshold vs FPR vs TPR",
        scene=dict(
            xaxis_title='Threshold',
            yaxis_title='False Positive Rate',
            zaxis_title='True Positive Rate',
            bgcolor='rgb(10,10,10)'
        )
    )
    fig_roc = go.Figure(data=[trace2], layout=layout2)
    fig_roc.write_html(os.path.join(output_dir, "roc_3d_surface.html"))
    fig_roc.write_image(os.path.join(output_dir, "roc_3d_surface.png"))

    print(f"Evaluation completed. Results saved to '{output_dir}'")


if __name__ == "__main__":
    evaluate_face_detector("/Volumes/main/Capstone/Clean/db/images_with_filters")