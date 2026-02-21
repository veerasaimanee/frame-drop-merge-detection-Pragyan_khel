# VTED v2.2 — Strict Forensic Temporal Integrity Validator

## 🔬 Project Overview
VTED (Video Temporal Error Detector) is a professional-grade forensic tool designed to validate the temporal and structural integrity of broadcast video streams. Specifically optimized for high-motion sports footage (e.g., cricket broadcast), VTED v2.2 employs a **deterministic, rule-based engine** to detect Frame Drops and Frame Merges with mathematical precision.

## 🚀 Key Features (v2.2)
- **Strict Forensic Logic:** No heuristic guesswork. Labels are assigned based on immutable timing and signal processing rules.
- **Robust Statistical Engine:** Implements **Median Absolute Deviation (MAD)** based Z-scores with a 21-frame rolling window to handle dynamic broadcast backgrounds.
- **Deep Structural Analysis:** Detects "ghosting" or blended frames (FRAME_MERGE) by correlating SSIM drops with Laplacian edge-blur verification.
- **Timestamp Audit:** Compares Metadata FPS against physical stream duration to compute a Validation FPS, flagging discrepancies.
- **Visual Evidence Engine:** Automatically generates forensic snapshots of flagged frames with data overlays (Frame index, Status, Confidence).
- **Professional Dark Dashboard:** A high-contrast, information-dense interface built for forensic specialists.
- **Automated Reporting:** Generates comprehensive PDF Integrity Reports and full Forensic CSV audit trails.

## ⚖️ Detection Methodology
### 1. FRAME_DROP (Rule 2.1)
Triggered when the inter-frame time difference is **>= 1.5x** the expected interval (based on metadata FPS). This provides a deterministic verdict on missing temporal packets.

### 2. FRAME_MERGE (Balanced Rule)
Triggered when timing is stable (±10% deviation) and structural signals converge via a two-layer threshold system:
- **Layer 1 (Strong):** SSIM Z-score < -2.3 AND Laplacian Z-score < -2.3.
- **Layer 2 (Moderate):** SSIM Z-score < -2.0 AND Laplacian Z-score < -1.8 AND local texture confirmation (Laplacian < 80% of local mean).

### 3. Structural Drop Fallback (Rule 6.1)
Detects drops even when timestamps are spoofed or constant:
- **Motion Z-score > 3.5** AND **SSIM Z-score < -3.0**

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
- Python 3.13
- OpenCV (cv2)
- NumPy
- Pandas
- Scikit-Image
- CustomTkinter
- fpdf2
- Pillow

---
**Disclaimer:** This tool is designed for forensic validation. Analysis parameters are tuned for broadcast-grade footage. Low-quality or highly compressed web video may require manual threshold adjustment in `video_analyzer.py`.

**Version:** 2.2 Strict Forensic
