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
        self.fps = 0
        self.expected_interval = 0
        self.window_size = 15 # Rolling window for local stats
        self.processing_metadata = {}
        self.fps_stats = {}

    def _robust_z_score(self, val, median, mad):
        """Calculate robust Z-score using Median Absolute Deviation."""
        return (val - median) / (1.4826 * mad + 1e-6)

    def _sigmoid(self, x):
        """Sigmoid function for confidence calibration."""
        return 1 / (1 + np.exp(-x))

    def collect_metrics(self, progress_callback=None):
        start_time = time.time()
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise Exception("Cannot open video file.")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            raise Exception("FPS could not be determined.")

        self.expected_interval = 1.0 / self.fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Debug metadata print
        print(f"Frame Count: {total_frames}")
        print(f"FPS: {self.fps}")
        print(f"Duration (sec): {total_frames / self.fps if self.fps > 0 else 0:.2f}")

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            raise Exception("Cannot read first frame.")

        # Optimization: One conversion, reuse grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_timestamp = 0.0
        frame_index = 1

        # Preallocate metrics list with estimated size to avoid frequent resizing
        self.metrics = []
        
        # Row 1 Placeholder
        self.metrics.append({
            "frame": 1,
            "timestamp": 0.0,
            "time_diff": 0.0,
            "motion_score": 0.0,
            "ssim": 1.0,
            "laplacian_var": cv2.Laplacian(prev_gray, cv2.CV_64F).var()
        })

        time_deltas = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            # Compute true timestamp based on capture if possible, 
            # but for standard video we use frame/fps or the position
            current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if current_timestamp == 0 and frame_index > 1:
                current_timestamp = (frame_index - 1) / self.fps
            
            time_diff = current_timestamp - prev_timestamp
            time_deltas.append(time_diff)

            # Grayscale conversion once
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Motion (Abs Difference)
            diff = cv2.absdiff(gray, prev_gray)
            motion_score = np.mean(diff)

            # SSIM
            ssim_score, _ = ssim(gray, prev_gray, full=True)

            # Laplacian (Texture)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            self.metrics.append({
                "frame": frame_index,
                "timestamp": current_timestamp,
                "time_diff": time_diff,
                "motion_score": motion_score,
                "ssim": ssim_score,
                "laplacian_var": lap_var
            })

            # Re-use current as previous for next iteration (no copy needed)
            prev_gray = gray
            prev_timestamp = current_timestamp
            
            if progress_callback and frame_index % 10 == 0:
                progress_callback(frame_index, total_frames)

        cap.release()
        
        duration = time.time() - start_time
        self.processing_metadata['runtime'] = duration
        self.processing_metadata['fps_processed'] = frame_index / duration if duration > 0 else 0
        self.processing_metadata['total_frames'] = frame_index

        # SECTION 1: DEFENSIBLE FPS MODEL
        if time_deltas:
            first_ts = self.metrics[0]['timestamp']
            last_ts = self.metrics[-1]['timestamp']
            total_duration = last_ts - first_ts
            computed_fps = (frame_index - 1) / total_duration if total_duration > 0 else self.fps
            
            self.fps_stats = {
                "metadata_fps": self.fps,
                "computed_true_fps": computed_fps,
                "total_frames": frame_index,
                "duration_seconds": total_duration,
                "mean_interval": np.mean(time_deltas),
                "std_interval": np.std(time_deltas)
            }
        
        return self.metrics

    def classify_frames(self):
        df = pd.DataFrame(self.metrics)
        
        # SECTION 7: EDGE CASE HANDLING - Frames < 10
        if len(df) < 10:
            print("Warning: Insufficient data for reliable anomaly detection. (Frames < 10)")
            df['status'] = "NORMAL"
            # Ensure required columns for reporting exist
            required_cols = ['motion_z', 'ssim_z', 'lap_z', 'anomaly_score', 'confidence', 'reason']
            for col in required_cols:
                df[col] = 0.0
            self.results = df
            return self.results, {"warning": "Insufficient data"}

        # 1. Temporal Model Correction (SECTION 1)
        # Exclude frame 1 for stats
        stats_df = df.iloc[1:].copy()
        
        time_vals = stats_df["time_diff"].values
        expected_interval = np.median(time_vals)
        self.expected_interval = expected_interval # Update class var
        temporal_tolerance = expected_interval * 0.2
        
        # 2. Metric Stats (Robust Z-Scores)
        motion_vals = stats_df["motion_score"].values
        ssim_vals = stats_df["ssim"].values
        lap_vals = stats_df["laplacian_var"].values

        m_med = np.median(motion_vals)
        m_mad = np.median(np.abs(motion_vals - m_med))
        
        s_med = np.median(ssim_vals)
        s_mad = np.median(np.abs(ssim_vals - s_med))
        
        l_med = np.median(lap_vals)
        l_mad = np.median(np.abs(lap_vals - l_med))

        # SECTION 7: EDGE CASE HANDLING - Duration < 0.3s
        video_duration = df['timestamp'].iloc[-1]
        merge_detection_enabled = video_duration >= 0.3

        # Prepare rolling stats for visualization plots (not used for classification)
        df['local_motion_mean'] = df['motion_score'].rolling(window=15, center=True).mean().fillna(m_med)
        df['local_ssim_mean'] = df['ssim'].rolling(window=15, center=True).mean().fillna(s_med)

        results = []
        
        for idx, row in df.iterrows():
            if row['frame'] == 1:
                results.append({
                    **row.to_dict(), 
                    "status": "NORMAL", 
                    "anomaly_score": 0.0, 
                    "confidence": 0.0, 
                    "reason": "Baseline", 
                    "motion_z": 0.0, 
                    "ssim_z": 0.0, 
                    "lap_z": 0.0
                })
                continue

            # Robust Z-Scores
            m_z = self._robust_z_score(row['motion_score'], m_med, m_mad)
            s_z = self._robust_z_score(row['ssim'], s_med, s_mad)
            l_z = self._robust_z_score(row['laplacian_var'], l_med, l_mad)
            
            # Simple scoring for reporting (not used for label)
            # Normalize deviations for score visualization
            m_score = min(1.0, max(0, m_z) / 5.0)
            s_score = min(1.0, max(0, -s_z) / 5.0)
            t_score = min(1.0, abs(row['time_diff'] - expected_interval) / (expected_interval * 1.5))
            anomaly_score = (t_score * 0.5) + (s_score * 0.3) + (m_score * 0.2)
            
            # SECTION 4 & 5: MUTUAL EXCLUSIVITY & DETERMINISTIC CLASSIFICATION
            status = "NORMAL"
            reason = "Consistent"

            # SECTION 2: FRAME DROP CORRECTION
            # time_diff > expected * 1.5 AND motion_z > 2.0
            if row['time_diff'] > expected_interval * 1.5 and m_z > 2.0:
                status = "FRAME_DROP"
                reason = "Temporal Anomaly (High Time Diff + Motion)"
            
            # SECTION 3: FRAME MERGE CORRECTION
            # ONLY if Drop condition not met
            elif merge_detection_enabled:
                # abs(time_diff - expected) <= tolerance AND ssim_z < -2.0 AND laplacian_z < -1.5 AND motion_z < 3.0
                if (abs(row['time_diff'] - expected_interval) <= temporal_tolerance and 
                    s_z < -2.0 and 
                    l_z < -1.5 and 
                    m_z < 3.0):
                    status = "FRAME_MERGE"
                    reason = "Structural Anomaly (Blur + Low SSIM)"

            results.append({
                **row.to_dict(),
                "status": status,
                "reason": reason,
                "motion_z": round(float(m_z), 2),
                "ssim_z": round(float(s_z), 2),
                "lap_z": round(float(l_z), 2),
                "anomaly_score": round(float(anomaly_score), 4),
                "confidence": 1.0 if status != "NORMAL" else 0.0 # Simple deterministic confidence
            })

        self.results = pd.DataFrame(results)
        
        # Summary Calculation
        total = len(self.results)
        drops = len(self.results[self.results['status'] == "FRAME_DROP"])
        merges = len(self.results[self.results['status'] == "FRAME_MERGE"])
        
        self.summary = {
            'video_name': os.path.basename(self.video_path),
            'total_frames': total,
            'fps': self.fps,
            'expected_interval': expected_interval,
            'drop_count': drops,
            'merge_count': merges,
            'drop_percent': (drops / total) * 100 if total > 0 else 0,
            'merge_percent': (merges / total) * 100 if total > 0 else 0,
            'integrity_score': max(0, 100 - ((drops + merges) / total * 100)) if total > 0 else 0,
            'mean_motion': motion_vals.mean(),
            'std_motion': motion_vals.std(),
            'mean_ssim': ssim_vals.mean(),
            'std_ssim': ssim_vals.std(),
            'm_med': m_med,
            'm_mad': m_mad,
            's_med': s_med,
            's_mad': s_mad,
            'merge_detection_enabled': merge_detection_enabled,
            'fps_stats': self.fps_stats
        }
        
        return self.results, self.summary

    @staticmethod
    def generate_synthetic_drop(video_path, output_path, drop_frame_index=30):
        """Removes a frame at chosen index to simulate a drop."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            if count == drop_frame_index:
                continue # Skip this frame
            out.write(frame)
            
        cap.release()
        out.release()
        return output_path

    @staticmethod
    def generate_synthetic_merge(video_path, output_path, merge_index=30):
        """Blends frames i and i+1 to simulate a merge/ghosting."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            if count == merge_index:
                ret2, next_frame = cap.read()
                if ret2:
                    merged = cv2.addWeighted(frame, 0.5, next_frame, 0.5, 0)
                    out.write(merged)
                    count += 1 # We consumed two frames but output one? 
                    # Actually user said: "Replace frame_i with merged". 
                    # If we output 'merged' and then continue, we've effectively replaced frame_i with the blend of i and i+1.
                    # Wait, if we replace frame_i with merged, do we still output frame_i+1?
                    # The user says: "Replace frame_i with merged ... Write new video".
                    # Usually a merge is one frame representing two.
                else:
                    out.write(frame)
            else:
                out.write(frame)
                
        cap.release()
        out.release()
        return output_path