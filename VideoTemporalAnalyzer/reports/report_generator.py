import os
import pandas as pd
from datetime import datetime
import cv2
import numpy as np
from fpdf import FPDF

class ReportGenerator:
    def __init__(self, output_base_dir="results"):
        self.output_base_dir = os.path.abspath(output_base_dir)
        if not os.path.exists(self.output_base_dir):
            os.makedirs(self.output_base_dir)

    def generate_reports(self, results_df, summary, video_path, custom_name=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.basename(video_path)
        base_name = os.path.splitext(video_name)[0]
        
        display_name = custom_name if custom_name and custom_name.strip() else base_name
        folder_name = f"VTED_RESULT_{display_name}_{timestamp}"
        output_dir = os.path.join(self.output_base_dir, folder_name)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # 1. Forensic CSV (Full Metrics - Rule 10.2)
        csv_path = os.path.join(output_dir, "frame_classification.csv")
        results_df.to_csv(csv_path, index=False)
        
        # 2. Anomaly Images (Rule 9.1)
        anomaly_dir = os.path.join(output_dir, "anomaly_images")
        flagged_images = self._generate_anomaly_snapshots(results_df, video_path, anomaly_dir)
        
        # 3. Forensic Analysis Log (Rule 10.2)
        log_path = self._generate_analysis_log(results_df, summary, output_dir)
        
        # 4. PDF Report (Rule 13.1)
        pdf_path = self._generate_pdf(summary, results_df, output_dir, flagged_images)
            
        return os.path.abspath(output_dir), os.path.abspath(csv_path), os.path.abspath(pdf_path)

    def _generate_anomaly_snapshots(self, df, video_path, output_dir):
        """Saves snapshots for every FRAME_DROP or FRAME_MERGE."""
        flagged = df[df['status'] != 'NORMAL']
        if flagged.empty:
            return []
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        image_paths = []
        cap = cv2.VideoCapture(video_path)
        
        for idx, row in flagged.iterrows():
            frame_idx = int(row['frame'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Create Dark Overlay for forensic info (Visual Branding)
            overlay = frame.copy()
            h, w = frame.shape[:2]
            cv2.rectangle(overlay, (0, 0), (min(w, 450), 160), (13, 17, 23), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Status Color Branding
            status_color = (46, 46, 255) if row['status'] == 'FRAME_DROP' else (159, 255, 0)
            
            text_lines = [
                f"STRICT VALIDATION: {row['status']}",
                f"Frame: {frame_idx} | TS: {row['timestamp']:.3f}s",
                f"Confidence: {row['confidence']*100:.1f}%",
                f"Temporal Dev: {row['temporal_deviation']:.2f}x",
                f"SSIM Z: {row['ssim_z']:.2f} | Lap Z: {row['laplacian_z']:.2f}"
            ]
            
            for i, line in enumerate(text_lines):
                color = status_color if i == 0 else (255, 255, 255)
                cv2.putText(frame, line, (15, 35 + i*28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            img_name = f"frame_{frame_idx:04d}_{row['status']}.png"
            img_path = os.path.join(output_dir, img_name)
            cv2.imwrite(img_path, frame)
            image_paths.append(os.path.abspath(img_path))
            
        cap.release()
        return image_paths

    def _generate_analysis_log(self, df, summary, output_dir):
        """Strict Rule Verification Log."""
        log_content = [
            "=== VTED v2.2 STRICT FORENSIC LOG ===",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Target: {summary['video_name']}",
            "-" * 50,
            "PRIMARY VERIFICATION BOUNDS:",
            f"Metadata FPS: {summary['metadata_fps']:.4f}",
            f"Computed FPS: {summary['computed_fps']:.4f}",
            f"Expected Frame Interval: {1.0/summary['metadata_fps'] if summary['metadata_fps']>0 else 0:.6f}s",
            f"Strict Drop Threshold: deviation >= 1.5",
            f"Strict Merge Threshold: dev ~1.0 AND ssim_z < -2.5 AND lap_z < -2.5",
            "-" * 50,
            f"TOTAL ANOMALIES DETECTED: {summary['drop_count'] + summary['merge_count']}",
            f"Frame Drops: {summary['drop_count']}",
            f"Frame Merges: {summary['merge_count']}",
            f"Temporal Integrity: {summary['integrity_score']:.2f}%",
            "-" * 50,
            "DETECTION REASONING LOG:"
        ]
        
        anomalies = df[df['status'] != 'NORMAL'].head(30)
        for _, row in anomalies.iterrows():
            log_content.append(f"• Frame {int(row['frame']):04d}: {row['status']} | Rule: {row['rule_info']} | Conf: {row['confidence']*100:.1f}%")
            
        log_path = os.path.join(output_dir, "analysis_log.txt")
        with open(log_path, "w") as f:
            f.write("\n".join(log_content))
        return os.path.abspath(log_path)

    def _generate_pdf(self, summary, results_df, output_dir, flagged_images):
        pdf = FPDF()
        pdf.add_page()
        
        # Branding Header
        pdf.set_fill_color(13, 17, 23)
        pdf.rect(0, 0, 210, 40, 'F')
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 24)
        pdf.cell(0, 20, "VTED Forensic Integrity Report", ln=True, align='C')
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, "Strict Forensic Temporal Integrity Validator v2.2", ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "1. Executive Summary", ln=True)
        pdf.set_font("Helvetica", "", 11)
        
        summary_lines = [
            f"Target File: {summary['video_name']}",
            f"Session Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Frames Analyzed: {summary['total_frames']}",
            f"Temporal Integrity Score: {summary['integrity_score']:.2f}%"
        ]
        for line in summary_lines:
            pdf.cell(0, 8, line, ln=True)
            
        pdf.ln(5)
        
        # Methodology (Rule 13.1)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "2. Forensic Methodology", ln=True)
        pdf.set_font("Helvetica", "", 10)
        methodology = (
            "This system employs a deterministic rule-based engine to validate frame sequence integrity. "
            "Timing anomalies (DROPS) are identified via precise inter-frame interval deviation analysis (1.5x threshold). "
            "Structural anomalies (MERGES) are validated using robust Z-scores based on Median Absolute Deviation (MAD) "
            "across 21-frame windows to ensure stability in sports footage. Multiple signal convergence "
            "(SSIM drop + Laplacian blur) is mandatory for any merge classification."
        )
        pdf.multi_cell(0, 6, methodology)
        
        pdf.ln(5)
        
        # FPS Consistency
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "3. System Metrics", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Metadata Reference FPS: {summary['metadata_fps']:.2f}", ln=True)
        pdf.cell(0, 8, f"Measured Stream Duration: {summary['duration']:.3f}s", ln=True)
        pdf.cell(0, 8, f"Frame Drop Count: {summary['drop_count']}", ln=True)
        pdf.cell(0, 8, f"Frame Merge Count: {summary['merge_count']}", ln=True)
        
        pdf.ln(10)
        
        # Evidence Samples
        if flagged_images:
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, "4. Forensic Evidence Snapshots", ln=True)
            for img in flagged_images[:5]: # Show top 5 in PDF
                if os.path.exists(img):
                    pdf.image(img, w=170)
                    pdf.ln(5)
        
        pdf_path = os.path.join(output_dir, "Forensic_Integrity_Report.pdf")
        pdf.output(pdf_path)
        return os.path.abspath(pdf_path)
