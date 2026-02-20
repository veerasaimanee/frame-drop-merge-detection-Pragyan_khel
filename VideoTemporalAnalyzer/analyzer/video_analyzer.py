import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import os
import time

class VideoAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.metrics = []
        self.results = None
        self.summary = {}
        self.metadata_fps = 0
        self.computed_fps = 0
        self.expected_interval = 0
        self.window_size = 21 # Forensic window for sports footage
        self.processing_metadata = {}

    def _robust_z_score(self, val, median, mad):
        """Standard Forensic MAD-based Z-score."""
        return (val - median) / (1.4826 * mad + 1e-9)

    def _sigmoid(self, x):
        """Sigmoid for confidence calibration."""
        return 1 / (1 + np.exp(-x))

    def collect_metrics(self, progress_callback=None):
        start_time = time.time()
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise Exception("Cannot open video file.")

        # 1. FPS Model (Strict)
        self.metadata_fps = cap.get(cv2.CAP_PROP_FPS)
        if self.metadata_fps <= 0:
             # Fallback if metadata is broken, but we log it
             self.metadata_fps = 30.0 
        
        self.expected_interval = 1.0 / self.metadata_fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            raise Exception("Cannot read first frame.")

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_timestamp = 0.0
        frame_index = 1

        self.metrics = []
        
        # Row 1 Baseline
        self.metrics.append({
            "frame": 1,
            "timestamp": 0.0,
            "time_diff": 0.0,
            "temporal_deviation": 1.0,
            "motion": 0.0,
            "ssim": 1.0,
            "laplacian": float(cv2.Laplacian(prev_gray, cv2.CV_64F).var())
        })

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            
            # Timestamp Retrieval (Standard Position)
            current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if current_timestamp == 0 and frame_index > 1:
                current_timestamp = (frame_index - 1) / self.metadata_fps
            
            time_diff = current_timestamp - prev_timestamp
            temporal_deviation = time_diff / self.expected_interval

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Motion (Mean Absolute Difference)
            diff = cv2.absdiff(gray, prev_gray)
            motion_score = float(np.mean(diff))

            # SSIM
            ssim_score, _ = ssim(gray, prev_gray, full=True)

            # Laplacian (Sharpness)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            self.metrics.append({
                "frame": frame_index,
                "timestamp": current_timestamp,
                "time_diff": time_diff,
                "temporal_deviation": temporal_deviation,
                "motion": motion_score,
                "ssim": float(ssim_score),
                "laplacian": lap_var
            })

            prev_gray = gray
            prev_timestamp = current_timestamp
            
            if progress_callback and frame_index % 10 == 0:
                progress_callback(frame_index, total_frames)

        cap.release()
        
        duration = time.time() - start_time
        self.processing_metadata['runtime'] = duration
        self.processing_metadata['total_frames'] = frame_index
        
        # Computed FPS Validation
        total_duration_ts = self.metrics[-1]['timestamp'] if self.metrics else 0
        self.computed_fps = (frame_index / total_duration_ts) if total_duration_ts > 0 else 0
        
        return self.metrics

    def classify_frames(self):
        df = pd.DataFrame(self.metrics)
        
        if len(df) < self.window_size:
            # Fallback for very short videos
            df['status'] = "NORMAL"
            df['confidence'] = 0.0
            for col in ['motion_z', 'ssim_z', 'laplacian_z']:
                df[col] = 0.0
            self.results = df
            return self.results, {"warning": "Insufficient context"}

        # Compute Rolling Stats for Forensic Z-Scores (Mandatory)
        # Using a window of 21 for stabilizing sports footage
        
        def get_rolling_metrics(series):
            # Rolling median and MAD
            rol = series.rolling(window=self.window_size, center=True)
            med = rol.median().fillna(method='bfill').fillna(method='ffill')
            def get_mad(x):
                m = np.median(x)
                return np.median(np.abs(x - m))
            mad = rol.apply(get_mad, raw=True).fillna(method='bfill').fillna(method='ffill')
            return med, mad

        motion_med, motion_mad = get_rolling_metrics(df['motion'])
        ssim_med, ssim_mad = get_rolling_metrics(df['ssim'])
        lap_med, lap_mad = get_rolling_metrics(df['laplacian'])

        results = []
        for idx, row in df.iterrows():
            if row['frame'] == 1:
                results.append({**row.to_dict(), "status": "NORMAL", "confidence": 0.0, "motion_z": 0.0, "ssim_z": 0.0, "laplacian_z": 0.0})
                continue

            # Z-Scores
            m_z = self._robust_z_score(row['motion'], motion_med.iloc[idx], motion_mad.iloc[idx])
            s_z = self._robust_z_score(row['ssim'], ssim_med.iloc[idx], ssim_mad.iloc[idx])
            l_z = self._robust_z_score(row['laplacian'], lap_med.iloc[idx], lap_mad.iloc[idx])

            # Classification Logic (Priority: Drop > Merge > Structural Drop > Normal)
            status = "NORMAL"
            rule_triggered = "N/A"

            # 1. Strict Temporal Drop
            if row['temporal_deviation'] >= 1.5:
                status = "FRAME_DROP"
                rule_triggered = "Temporal Anomaly (Rule 2.1)"
            
            # 2. Strict Frame Merge
            elif (abs(row['temporal_deviation'] - 1.0) <= 0.1 and 
                  s_z < -2.5 and 
                  l_z < -2.5):
                status = "FRAME_MERGE"
                rule_triggered = "Structural Merge (Rule 4.1)"
            
            # 3. Structural Drop Fallback
            elif (m_z > 3.5 and s_z < -3.0):
                status = "FRAME_DROP"
                rule_triggered = "Structural Drop (Rule 6.1)"

            # Confidence Calculation (Rule 8.1)
            raw_conf = (0.6 * abs(row['temporal_deviation'] - 1.0) + 
                        0.2 * abs(s_z) + 
                        0.2 * abs(l_z))
            confidence = float(self._sigmoid(raw_conf - 2.0)) # Centered around anomaly trigger

            results.append({
                **row.to_dict(),
                "status": status,
                "confidence": round(confidence, 4),
                "motion_z": round(m_z, 2),
                "ssim_z": round(s_z, 2),
                "laplacian_z": round(l_z, 2),
                "rule_info": rule_triggered
            })

        self.results = pd.DataFrame(results)
        
        # Summary Calculation
        total = len(self.results)
        drops = len(self.results[self.results['status'] == "FRAME_DROP"])
        merges = len(self.results[self.results['status'] == "FRAME_MERGE"])
        accuracy = 100 - ((drops + merges) / total * 100) if total > 0 else 0
        
        self.summary = {
            'video_name': os.path.basename(self.video_path),
            'total_frames': total,
            'metadata_fps': self.metadata_fps,
            'computed_fps': self.computed_fps,
            'drop_count': drops,
            'merge_count': merges,
            'integrity_score': accuracy,
            'duration': self.metrics[-1]['timestamp'] if self.metrics else 0
        }
        
        return self.results, self.summary