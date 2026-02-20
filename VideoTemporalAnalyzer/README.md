# Video Temporal Error Detector v2.0 – Forensic Frame Analysis System

## Project Overview
This application is a professional forensic tool designed to detect temporal inconsistencies in video streams. Version 2.0 introduces **Defensible FPS Modeling**, **Separated Forensic/Clean Outputs**, and **Evidence Image Generation** for Frame Drops and Merges.

## Key Features (v2.0)
- **Defensible FPS Model:** Computes true FPS based on inter-frame intervals, flagging metadata inconsistencies > 5%.
- **Clean vs Audit Metrics:**
  - `frame_classification.csv`: Clean report for non-technical review.
  - `frame_audit_metrics.csv`: Detailed forensic trace containing Z-scores and component contribution scores.
- **Evidence Image Generation:** Automatically overlays metadata on frames flagged as DROP or MERGE.
- **Side-by-Side Merge Validation:** Generates comparison images (Prev | Merge | Next) for ghosting analysis.
- **Professional PDF Reports:** Generates structured reports with embedded evidence samples using `fpdf2`.
- **Synthetic Test Tools:** Built-in generators to create controlled Frame Drop and Frame Merge scenarios for system validation.
- **Custom Naming & Isolation:** Every analysis creates a unique, timestamped session folder.

## Detection Logic (Tightened)
- **FRAME_DROP:** Triggered only if `Time Delta > 1.5x Interval` **AND** `Motion Z-Score > 2.0` **AND** `SSIM Z-Score < -1.5`.
- **FRAME_MERGE:** Triggered only if `Timing is Stable` **AND** `SSIM < local_threshold` **AND** `Laplacian Texture < median` **AND** `Motion is not extreme`.

## System Architecture
```text
VideoTemporalAnalyzer/
├── main.py                 # Application Entry Point
├── analyzer/               # Core Logic
│     └── video_analyzer.py # Pass 1 & Pass 2 + Synthetic Tools
├── reports/                # Output Generation
│     └── report_generator.py # CSV/PDF/Evidence Generation
├── gui/                    # Presentation Layer
│     └── app.py            # CustomTkinter GUI Implementation
├── results/                # Session-Isolated Output Storage
├── requirements.txt        # Updated Dependencies (fpdf2, Pillow)
└── README.md               # Documentation
```

## How to Run
1.  **Environment Setup:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Launch Application:**
    ```bash
    python main.py
    ```
3.  **Process Video:**
    - **Step 1:** Select a video file.
    - **Custom Title:** Enter a custom name for the analysis session.
    - **Step 2:** Click **Run Analysis**.
    - **Export:** Use the bottom bar to export PDF, CSV, or a full ZIP archive of the forensic folder.

## Synthetic Testing
Use the **"Gen Synthetic DROP/MERGE"** buttons in the bottom right to create test videos. The system should correctly identify these simulated errors with high confidence.

---
**Author:** Antigravity (Advanced Agentic Coding AI)
**Tech Stack:** Python 3.13, OpenCV, NumPy, Scikit-Image, Pandas, CustomTkinter, fpdf2.
