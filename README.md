# VTED v2.5 — Strict Forensic Temporal Integrity Validator

## 🔬 Project Overview
VTED (Video Temporal Error Detector) is a professional-grade forensic tool designed to validate the temporal and structural integrity of broadcast video streams. Specifically optimized for high-motion sports footage (e.g., cricket broadcast), VTED v2.5 employs a **deterministic, rule-based engine** to detect Frame Drops and Frame Merges with mathematical precision.

## 🚀 Key Features (v2.5)
- **Strict Forensic Logic:** No heuristic guesswork. Labels are assigned based on immutable timing and signal processing rules.
- **Robust Statistical Engine:** Implements **Median Absolute Deviation (MAD)** based Z-scores for reliable outlier detection.
- **Deep Structural Analysis:** Detects "ghosting" or blended frames (FRAME_MERGE) using SSIM and Laplacian variance.
- **Visual Evidence Engine:** Automatically captures forensic snapshots of flagged frames with data overlays.
- **Automated Reporting:** Generates comprehensive PDF Integrity Reports and Forensic CSV audit trails.


## 🔬 Visual Evidence Examples
VTED automatically captures and annotates frames where temporal anomalies are detected.

| Frame Drop Detection | Frame Merge Detection |
| :---: | :---: |
| ![Frame Drop](assets/example_frame_drop.png) | ![Frame Merge](assets/example_frame_merge.png) |
| *Rule 2.1: Temporal Gap Detected* | *Rule Layer 1: Structural Blend Detected* |

## ⚖️ Detection Methodology
### 1. FRAME_DROP (Rule 2.1)
Triggered when the inter-frame time difference is **>= 1.5x** the expected interval (based on metadata FPS). This provides a deterministic verdict on missing temporal packets.

### 2. FRAME_MERGE (Balanced Rule)
Triggered when timing is stable (±10% deviation) and structural signals converge via a two-layer threshold system:
- **Layer 1 (Strong):** SSIM Z-score < -2.3 AND Laplacian Z-score < -2.3.
- **Layer 2 (Moderate):** SSIM Z-score < -2.0 AND Laplacian Z-score < -1.8 AND local texture confirmation.

## 📂 Project Structure
```text
VideoTemporalAnalyzer/
├── main.py                 # Application Entry Point
├── analyzer/               # Strict Forensic Engine
│     └── video_analyzer.py # Timing & Statistical Math (Z-Scores, MAD)
├── gui/                    # Presentation Layer
│     └── app.py            # Professional Dark-Mode Dashboard
├── reports/                # Forensic Output Logic
│     └── report_generator.py # PDF/CSV Evidence Generation
├── assets/                 # Documentation Assets (Screenshots)
├── results/                # Session-Isolated Output Storage (Git Ignored)
├── requirements.txt        # System Dependencies
└── README.md               # Documentation
```

## 🛠️ Installation & Usage
1.  **Clone & Setup:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Launch:**
    ```bash
    python main.py
    ```
3.  **Validate:**
    - Select your video file.
    - Click **Run Analysis**.
    - Review results in the **Right Panel**.
    - Export the **Forensic Integrity Report (PDF)** for professional documentation.

## 📝 Dependencies
- **Python 3.13**
- **OpenCV (cv2)**: Video processing
- **NumPy & Pandas**: Data synthesis and analysis
- **Scikit-Image**: SSIM computation
- **CustomTkinter**: Modern GUI components
- **fpdf2**: PDF report generation
- **Pillow**: Image processing for overlays

---
**Disclaimer:** This tool is designed for forensic validation. Analysis parameters are tuned for broadcast-grade footage.
**Version:** 2.5 Strict Forensic
