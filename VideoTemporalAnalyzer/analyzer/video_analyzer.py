import cv2
import numpy as np
import pandas as pd
import os
from skimage.metrics import structural_similarity as ssim


class VideoAnalyzer:
    def __init__(self, video_path):
        self.video_path = os.path.abspath(video_path)
        self.frames_data = []
        self.metadata_fps = 0
        self.expected_interval = 0
        self.total_frames = 0
        self.duration = 0
        self.computed_fps = 0

    # ---------------------------
    # METRIC COLLECTION
    # ---------------------------

    def collect_metrics(self, progress_callback=None):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        self.metadata_fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.metadata_fps <= 0:
            self.metadata_fps = 30.0

        self.expected_interval = 1.0 / self.metadata_fps

        prev_gray = None
        first_ts = None
        last_ts = None

        for i in range(self.total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            if i == 0:
                first_ts = ts
                time_diff = self.expected_interval
            else:
                time_diff = ts - last_ts

            temporal_deviation = (
                time_diff / self.expected_interval
                if self.expected_interval > 0
                else 1.0
            )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            laplacian_raw = cv2.Laplacian(gray, cv2.CV_64F).var()
            ssim_raw = 1.0
            motion_raw = 0.0

            if prev_gray is not None:
                ssim_raw = ssim(prev_gray, gray, win_size=7)
                motion_raw = np.mean(cv2.absdiff(prev_gray, gray))

            self.frames_data.append(
                {
                    "frame_index": i,
                    "timestamp": ts,
                    "time_diff": time_diff,
                    "temporal_deviation": temporal_deviation,
                    "ssim_raw": ssim_raw,
                    "laplacian_raw": laplacian_raw,
                    "motion_raw": motion_raw,
                }
            )

            prev_gray = gray
            last_ts = ts

            if progress_callback:
                progress_callback(i + 1, self.total_frames)

        cap.release()

        if len(self.frames_data) > 1:
            self.duration = (
                self.frames_data[-1]["timestamp"]
                - self.frames_data[0]["timestamp"]
            )
            if self.duration <= 0:
                self.duration = len(self.frames_data) / self.metadata_fps

            self.computed_fps = (
                len(self.frames_data) / self.duration
                if self.duration > 0
                else self.metadata_fps
            )
        else:
            self.duration = 0
            self.computed_fps = self.metadata_fps

    # ---------------------------
    # ROBUST Z SCORE
    # ---------------------------

    def _robust_z(self, values):
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        mad = max(mad, 1e-6)
        return (values - median) / (1.4826 * mad)

    # ---------------------------
    # CLASSIFICATION
    # ---------------------------

    def classify_frames(self):
        if not self.frames_data:
            return pd.DataFrame(), {}

        df = pd.DataFrame(self.frames_data)

        df["motion_z"] = self._robust_z(df["motion_raw"].values)
        df["ssim_z"] = self._robust_z(df["ssim_raw"].values)
        df["laplacian_z"] = self._robust_z(df["laplacian_raw"].values)

        median_lap = np.median(df["laplacian_raw"])

        results = []

        for i, row in df.iterrows():

            if i == 0:
                results.append({**row, "status": "NORMAL"})
                continue

            status = "NORMAL"

            # ---------------------------
            # 1️⃣ TEMPORAL DROP
            # ---------------------------
            is_temporal_drop = (
                row["temporal_deviation"] >= 1.6
                and (row["motion_z"] > 1.5 or row["ssim_z"] < -1.8)
            )

            # ---------------------------
            # 2️⃣ STRUCTURAL DROP
            # ---------------------------
            is_structural_drop = (
                row["motion_z"] > 2.8 and row["ssim_z"] < -2.5
            )

            # ---------------------------
            # 3️⃣ STRICT MERGE
            # ---------------------------
            is_strict_merge = (
                abs(row["temporal_deviation"] - 1.0) <= 0.15
                and row["ssim_z"] < -1.8
                and row["laplacian_z"] < -1.6
                and row["motion_z"] < 3.0
                and row["laplacian_raw"] < (median_lap * 0.85)
            )

            # ---------------------------
            # 4️⃣ DUPLICATE / AI BLEND
            # ---------------------------
            is_ai_blend = (
                row["ssim_raw"] > 0.995
                and row["motion_raw"] < 1.0
                and abs(row["temporal_deviation"] - 1.0) <= 0.1
            )

            # PRIORITY
            if is_temporal_drop:
                status = "FRAME_DROP"
            elif is_structural_drop:
                status = "FRAME_DROP"
            elif is_strict_merge or is_ai_blend:
                status = "FRAME_MERGE"

            results.append({**row, "status": status})

        results_df = pd.DataFrame(results)

        drop_count = len(results_df[results_df["status"] == "FRAME_DROP"])
        merge_count = len(results_df[results_df["status"] == "FRAME_MERGE"])

        integrity_score = (
            100
            - ((drop_count + merge_count) / len(results_df) * 100)
            if len(results_df) > 0
            else 100
        )

        summary = {
            "video_name": os.path.basename(self.video_path),
            "total_frames": len(results_df),
            "duration": self.duration,
            "metadata_fps": self.metadata_fps,
            "computed_fps": self.computed_fps,
            "drop_count": drop_count,
            "merge_count": merge_count,
            "integrity_score": round(integrity_score, 2),
            "report_description": self._get_detailed_report_description()
        }

        return results_df, summary

    def _get_detailed_report_description(self):
        """Spec v3.0 Summary."""
        return (
            "VTED v3.0 FORENSIC AUDIT\n"
            "--------------------------------------------------\n"
            "ENGINE: Global Robust Model (No Rolling Variance)\n"
            "PRIORITY:\n"
            "1. Temporal Drop (dev >= 1.6 + structural confirmation)\n"
            "2. Structural Drop (motion_z > 2.8, ssim_z < -2.5)\n"
            "3. Balanced Merge (Z-score + Absolute Texture Drop)\n"
            "4. AI Blend (Extremely High SSIM + Low Motion)\n\n"
            "STATUS: Integrity validated against global video distribution."
        )

    @staticmethod
    def generate_synthetic_drop(video_path, output_path, drop_frame_index=30):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            count += 1
            if count == drop_frame_index: continue
            out.write(frame)
        cap.release()
        out.release()
        return output_path

    @staticmethod
    def generate_synthetic_merge(video_path, output_path, merge_index=30):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            count += 1
            if count == merge_index:
                ret2, next_frame = cap.read()
                if ret2:
                    merged = cv2.addWeighted(frame, 0.5, next_frame, 0.5, 0)
                    out.write(merged)
                    count += 1
                else: out.write(frame)
            else: out.write(frame)
        cap.release()
        out.release()
        return output_path
