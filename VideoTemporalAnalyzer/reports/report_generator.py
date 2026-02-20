import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import cv2
import numpy as np
from fpdf import FPDF

# Use non-interactive backend for server/thread safety
matplotlib.use('Agg')

class ReportGenerator:
    def __init__(self, output_base_dir="results"):
        self.output_base_dir = output_base_dir
        if not os.path.exists(self.output_base_dir):
            os.makedirs(self.output_base_dir)

    def generate_reports(self, results_df, summary, video_path, custom_name=None, analyzer_metadata=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.basename(video_path)
        base_name = custom_name if custom_name else os.path.splitext(video_name)[0]
        folder_name = f"{base_name}_{timestamp}"
        output_dir = os.path.join(self.output_base_dir, folder_name)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 1. Separate CSVs
        # Clean CSV
        clean_rename = {
            'frame': 'frame_index',
            'timestamp': 'timestamp_seconds',
            'time_diff': 'time_diff_seconds',
            'motion_score': 'motion_value',
            'ssim': 'ssim_value',
            'laplacian_var': 'laplacian_value',
            'status': 'status'
        }
        clean_cols = list(clean_rename.keys())
        clean_csv_path = os.path.join(output_dir, "frame_classification.csv")
        results_df[clean_cols].rename(columns=clean_rename).to_csv(clean_csv_path, index=False)
        
        # Audit CSV
        audit_exclude = ['local_motion_mean', 'local_ssim_mean']
        audit_cols = [c for c in results_df.columns if c not in audit_exclude]
        audit_csv_path = os.path.join(output_dir, "frame_audit_metrics.csv")
        results_df[audit_cols].to_csv(audit_csv_path, index=False)

        # FPS Details CSV
        fps_stats = summary.get('fps_stats', {})
        fps_df = pd.DataFrame([fps_stats])
        fps_csv_path = os.path.join(output_dir, "fps_details.csv")
        fps_df.to_csv(fps_csv_path, index=False)
        
        # 2. Visualization (PNG Plots)
        self._generate_plots(results_df, output_dir)
        
        # 3. Evidence Images
        flagged_dir = os.path.join(output_dir, f"{base_name}_flagged_frames")
        flagged_images = self._generate_evidence_images(results_df, video_path, flagged_dir)
        
        # 4. Professional Validation Log (analysis_log.txt)
        log_path = self._generate_validation_log(summary, analyzer_metadata, output_dir)
        
        # 5. Analysis Summary
        summary_text = self._generate_summary_text(summary, output_dir)
        
        # 6. PDF Report
        pdf_path = self._generate_pdf(summary, summary_text, output_dir, flagged_images)
            
        return output_dir, clean_csv_path, log_path, summary_text

    def _generate_plots(self, df, output_dir):
        """Generates forensic plots for visual audit."""
        frames = df['frame']

        # Plot 1: Motion Over Time
        plt.figure(figsize=(12, 4))
        plt.plot(frames, df['motion_score'], color='#3498db', label='Motion Magnitude')
        plt.fill_between(frames, df['local_motion_mean'], alpha=0.3, color='#3498db', label='Rolling Average')
        plt.title("Temporal Motion Profile")
        plt.xlabel("Frame Index")
        plt.ylabel("Magnitude (MAD)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "motion_plot.png"), dpi=150)
        plt.close()

        # Plot 2: SSIM Over Time
        plt.figure(figsize=(12, 4))
        plt.plot(frames, df['ssim'], color='#2ecc71', label='Structural Similarity')
        plt.axhline(y=df['ssim'].median(), color='red', linestyle='--', alpha=0.5, label='Median')
        plt.title("Structural Integrity (SSIM)")
        plt.xlabel("Frame Index")
        plt.ylabel("SSIM Score")
        plt.ylim(min(0.8, df['ssim'].min()), 1.05)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "ssim_plot.png"), dpi=150)
        plt.close()

        # Plot 3: Anomaly Score (Diagnostic Trace)
        plt.figure(figsize=(12, 4))
        plt.plot(frames, df['anomaly_score'], color='#e74c3c', label='Anomaly Score')
        plt.axhline(y=0.5, color='orange', linestyle=':', label='Threshold Band')
        plt.title("Diagnostic Anomaly Trace (0.0 - 1.0)")
        plt.xlabel("Frame Index")
        plt.ylabel("Anomaly Probability")
        plt.fill_between(frames, df['anomaly_score'], where=(df['anomaly_score'] >= 0.5), color='red', alpha=0.3)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "anomaly_plot.png"), dpi=150)
        plt.close()

    def _generate_validation_log(self, summary, metadata, output_dir):
        """Creates the engineering audit log."""
        log_content = [
            "=== PROFESSIONAL VALIDATION LOG v2.0 ===",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Video: {summary['video_name']}",
            "-" * 40,
            "SYSTEM PARAMETERS:",
            f"Window Size: 15 frames",
            f"Expected Interval: {1.0/summary['fps']:.6f}s",
            f"Global Motion Median: {summary['m_med']:.4f}",
            f"Global Motion MAD: {summary['m_mad']:.4f}",
            f"SSIM 2nd Percentile: {summary['ssim_threshold']:.4f}",
            "-" * 40,
            "PERFORMANCE METRICS:",
            f"Analysis Duration: {metadata.get('runtime', 0):.2f}s",
            f"Processing Speed: {metadata.get('fps_processed', 0):.1f} FPS",
            f"Total Frames Processed: {metadata.get('total_frames', 0)}",
            "-" * 40,
            "STATISTICAL BASELINES:",
            f"Mean Motion: {summary['mean_motion']:.6f}",
            f"Mean SSIM: {summary['mean_ssim']:.6f}",
            f"SSIM std: {summary['std_ssim']:.6f}"
        ]
        
        path = os.path.join(output_dir, "analysis_log.txt")
        with open(path, "w") as f:
            f.write("\n".join(log_content))
        return path

    def _generate_summary_text(self, summary, output_dir):
        summary_text = (
            f"Video Name: {os.path.splitext(summary['video_name'])[0]}\n"
            f"Total Frames: {summary['total_frames']}\n"
            f"FPS: {summary['fps']:.1f}\n"
            f"Drop Count: {summary['drop_count']}\n"
            f"Merge Count: {summary['merge_count']}\n"
            f"Temporal Integrity Score: {summary['integrity_score']:.2f}%\n\n"
            "DETECTION REASONING (Explainable v2.1):\n"
            "- FRAME_DROPS: Triggered by timing anomalies (> 1.5x median interval) and significant motion (> 2.0 Sigma).\n"
            "- FRAME_MERGE: Triggered by structural blur (SSIM/Laplacian shift) with stable timing.\n\n"
            "FORENSIC TRACEABILITY:\n"
            "- Detailed `frame_classification.csv` saved in results.\n"
            "- `analysis_log.txt` contains statistical parameters for audit.\n"
            "- Visualization plots (motion, ssim, anomaly) generated as PNGs.\n\n"
            "CONFIDENCE NOTES:\n"
            "- Scores are sigmoid-calibrated based on multi-factor weighted anomaly scores.\n"
            "- High Anomaly Score (>0.6) suggests definitive temporal glitch requiring review."
        )
        
        txt_path = os.path.join(output_dir, "analysis_summary.txt")
        with open(txt_path, "w") as f:
            f.write(summary_text)
        return summary_text
    def _generate_evidence_images(self, df, video_path, output_dir):
        flagged = df[df['status'].isin(['FRAME_DROP', 'FRAME_MERGE'])]
        if flagged.empty:
            return []
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        image_paths = []
        cap = cv2.VideoCapture(video_path)
        
        for idx, row in flagged.iterrows():
            frame_idx = int(row['frame'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Create overlay
            overlay = frame.copy()
            h, w = frame.shape[:2]
            cv2.rectangle(overlay, (0, 0), (min(w, 400), 120), (0, 0, 0), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            text_lines = [
                f"Video: {os.path.basename(video_path)}",
                f"Frame: {frame_idx}",
                f"Timestamp: {row['timestamp']:.3f}s",
                f"Status: {row['status']}"
            ]
            
            for i, line in enumerate(text_lines):
                cv2.putText(frame, line, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            if row['status'] == 'MERGE':
                # Generate side-by-side: Prev | Flagged | Next
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 2)
                ret_prev, prev_frame = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret_next, next_frame = cap.read()
                
                if ret_prev and ret_next:
                    # Resize for display
                    h_small = 360
                    w_small = int(w * (h_small / h))
                    prev_s = cv2.resize(prev_frame, (w_small, h_small))
                    curr_s = cv2.resize(frame, (w_small, h_small))
                    next_s = cv2.resize(next_frame, (w_small, h_small))
                    combined = np.hstack([prev_s, curr_s, next_s])
                    
                    img_name = f"frame_{frame_idx:04d}_{row['status']}_COMPARISON.png"
                    img_path = os.path.join(output_dir, img_name)
                    cv2.imwrite(img_path, combined)
                    image_paths.append(img_path)
            
            img_name = f"frame_{frame_idx:04d}_{row['status']}.png"
            img_path = os.path.join(output_dir, img_name)
            cv2.imwrite(img_path, frame)
            image_paths.append(img_path)
            
        cap.release()
        return image_paths

    def _generate_pdf(self, summary, summary_text, output_dir, flagged_images):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Video Temporal Integrity Report", ln=True, align='C')
        pdf.ln(5)
        
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(0, 10, summary_text)
        pdf.ln(5)
        
        # FPS Consistency Analysis
        fps_stats = summary.get('fps_stats', {})
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "FPS Consistency Analysis", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Metadata FPS: {fps_stats.get('metadata_fps', 0):.2f}", ln=True)
        pdf.cell(0, 8, f"Computed FPS (timestamp-based): {fps_stats.get('computed_true_fps', 0):.2f}", ln=True)
        pdf.cell(0, 8, f"Mean Frame Interval: {fps_stats.get('mean_interval', 0):.6f}s", ln=True)
        pdf.cell(0, 8, f"Interval Std Dev: {fps_stats.get('std_interval', 0):.6f}s", ln=True)
        
        m_fps = fps_stats.get('metadata_fps', 1)
        c_fps = fps_stats.get('computed_true_fps', 1)
        if abs(m_fps - c_fps) / m_fps > 0.05:
            pdf.set_text_color(255, 0, 0)
            pdf.cell(0, 10, "Metadata-FPS inconsistency detected.", ln=True)
            pdf.set_text_color(0, 0, 0)
        
        pdf.ln(10)
        
        # Anomalies
        if flagged_images:
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 10, "Flagged Evidence Samples (Up to 3)", ln=True)
            for img in flagged_images[:3]:
                # Add image, try to fit width
                pdf.image(img, w=180)
                pdf.ln(5)
        else:
            pdf.set_font("Helvetica", "I", 12)
            pdf.multi_cell(0, 10, "No statistically significant temporal anomalies detected.\nTiming variance and structural metrics remained within expected bounds.")

        pdf_path = os.path.join(output_dir, "analysis_report.pdf")
        pdf.output(pdf_path)
        return pdf_path
