import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import threading
import shutil
import cv2
from PIL import Image, ImageTk
from analyzer.video_analyzer import VideoAnalyzer
from reports.report_generator import ReportGenerator

# Strict Branding
D_BG = "#0D1117"
D_ACCENT = "#00AEEF"
D_GREEN = "#00FF9F"
D_RED = "#FF2E2E"
D_CARD = "#161B22"
D_TEXT = "#C9D1D9"

class VideoAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("VTED v2.2 — Strict Forensic Temporal Integrity Validator")
        self.geometry("1400x850")
        self.configure(fg_color=D_BG)
        
        self.video_path = None
        self.output_dir = None
        self.analyzer = None
        self.report_gen = ReportGenerator("results")
        self.current_frame_img = None

        self._setup_ui()

    def _setup_ui(self):
        self.grid_columnconfigure(1, weight=6) # Center Panel
        self.grid_columnconfigure(2, weight=3) # Right Panel
        self.grid_rowconfigure(0, weight=1)

        # 1. Left Panel (Actions)
        self.left_panel = ctk.CTkFrame(self, fg_color=D_CARD, corner_radius=0, width=220)
        self.left_panel.grid(row=0, column=0, sticky="nsew")
        self.left_panel.grid_propagate(False)

        ctk.CTkLabel(self.left_panel, text="VTED", font=ctk.CTkFont(size=24, weight="bold"), text_color=D_ACCENT).pack(pady=(30, 0))
        ctk.CTkLabel(self.left_panel, text="Forensic Engine v2.2", font=ctk.CTkFont(size=10), text_color=D_TEXT).pack(pady=(0, 40))

        self.upload_btn = self._create_nav_btn("Upload Video", self._upload_video)
        self.analyze_btn = self._create_nav_btn("Run Analysis", self._start_analysis, state="disabled", color=D_ACCENT)
        self.csv_btn = self._create_nav_btn("Download CSV", self._download_csv, state="disabled")
        self.pdf_btn = self._create_nav_btn("Download PDF", self._download_pdf, state="disabled")
        self.folder_btn = self._create_nav_btn("Results Folder", self._open_results, state="disabled")

        # 2. Center Panel (Video/Result Badge)
        self.center_panel = ctk.CTkFrame(self, fg_color=D_BG, corner_radius=0)
        self.center_panel.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        self.video_display = ctk.CTkLabel(self.center_panel, text="NO VIDEO LOADED", fg_color=D_CARD, corner_radius=15, 
                                          font=ctk.CTkFont(size=16), text_color=D_TEXT)
        self.video_display.pack(fill="both", expand=True, padx=10, pady=10)

        # Anomaly Status Badge (Rule 11)
        self.status_badge = ctk.CTkLabel(self.center_panel, text="SYSTEM READY", fg_color=D_CARD, corner_radius=20,
                                         height=50, width=300, font=ctk.CTkFont(size=18, weight="bold"), text_color=D_TEXT)
        self.status_badge.pack(pady=20)

        # 3. Right Panel (Stats)
        self.right_panel = ctk.CTkFrame(self, fg_color=D_CARD, corner_radius=15)
        self.right_panel.grid(row=0, column=2, sticky="nsew", padx=(0, 20), pady=20)
        
        ctk.CTkLabel(self.right_panel, text="Forensic Summary", font=ctk.CTkFont(size=18, weight="bold"), text_color=D_ACCENT).pack(pady=20)

        self.stats_labels = {}
        stats_keys = ["Video Name", "Total Frames", "Duration", "Metadata FPS", "Computed FPS", "Drop Count", "Merge Count", "Integrity Score"]
        
        for key in stats_keys:
            container = ctk.CTkFrame(self.right_panel, fg_color="transparent")
            container.pack(fill="x", padx=20, pady=5)
            ctk.CTkLabel(container, text=key, text_color=D_TEXT, font=ctk.CTkFont(size=12)).pack(side="left")
            val_label = ctk.CTkLabel(container, text="--", text_color="white", font=ctk.CTkFont(size=12, weight="bold"))
            val_label.pack(side="right")
            self.stats_labels[key] = val_label

        # Progress
        self.progress_bar = ctk.CTkProgressBar(self.right_panel, width=200, fg_color="#333", progress_color=D_ACCENT)
        self.progress_bar.pack(pady=(40, 10))
        self.progress_bar.set(0)
        self.status_text = ctk.CTkLabel(self.right_panel, text="Idle", font=ctk.CTkFont(size=11), text_color=D_TEXT)
        self.status_text.pack()

    def _create_nav_btn(self, text, command, state="normal", color=None):
        btn = ctk.CTkButton(self.left_panel, text=text, command=command, state=state,
                             fg_color=color if color else "transparent", 
                             border_width=1 if not color else 0,
                             border_color=D_ACCENT,
                             hover_color=D_CARD if not color else D_ACCENT,
                             height=40, font=ctk.CTkFont(size=13))
        btn.pack(fill="x", padx=20, pady=10)
        return btn

    def _upload_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov")])
        if path:
            self.video_path = path
            self.analyze_btn.configure(state="normal")
            filename = os.path.basename(path)
            self.stats_labels["Video Name"].configure(text=filename[:20] + ".." if len(filename) > 20 else filename)
            self.status_text.configure(text="Video Loaded")
            self._display_first_frame()

    def _display_first_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if ret:
            self._update_preview(frame)
        cap.release()

    def _update_preview(self, frame, is_anomaly=False):
        # Resize to fit display
        h, w = frame.shape[:2]
        display_w = 800
        display_h = int(h * (display_w / w))
        
        # Overlay Red if anomaly (Rule 11)
        if is_anomaly:
            red_tint = np.zeros_like(frame)
            red_tint[:, :] = (0, 0, 150) # Red in BGR
            frame = cv2.addWeighted(frame, 0.7, red_tint, 0.3, 0)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((display_w, display_h), Image.Resampling.LANCZOS)
        
        self.current_frame_img = ImageTk.PhotoImage(img)
        self.video_display.configure(image=self.current_frame_img, text="")

    def _start_analysis(self):
        self.analyze_btn.configure(state="disabled")
        self.upload_btn.configure(state="disabled")
        self.status_text.configure(text="Processing...")
        
        threading.Thread(target=self._run_analysis_thread, daemon=True).start()

    def _run_analysis_thread(self):
        try:
            self.analyzer = VideoAnalyzer(self.video_path)
            
            def cb(current, total):
                self.progress_bar.set(current / total)
            
            self.analyzer.collect_metrics(progress_callback=cb)
            results, summary = self.analyzer.classify_frames()
            
            self.output_dir, self.csv_path, self.pdf_path = self.report_gen.generate_reports(results, summary, self.video_path)
            
            self.after(0, lambda: self._update_ui_results(summary, results))
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))
            self.after(0, self._reset_ui)

    def _update_ui_results(self, summary, results):
        self.stats_labels["Total Frames"].configure(text=str(summary['total_frames']))
        self.stats_labels["Duration"].configure(text=f"{summary['duration']:.2f}s")
        self.stats_labels["Metadata FPS"].configure(text=f"{summary['metadata_fps']:.2f}")
        self.stats_labels["Computed FPS"].configure(text=f"{summary['computed_fps']:.2f}")
        self.stats_labels["Drop Count"].configure(text=str(summary['drop_count']), text_color=D_RED if summary['drop_count'] > 0 else "white")
        self.stats_labels["Merge Count"].configure(text=str(summary['merge_count']), text_color=D_GREEN if summary['merge_count'] > 0 else "white")
        self.stats_labels["Integrity Score"].configure(text=f"{summary['integrity_score']:.2f}%")

        # Update Badge
        if summary['drop_count'] > 0 or summary['merge_count'] > 0:
            self.status_badge.configure(text="! ANOMALIES DETECTED !", fg_color=D_RED, text_color="white")
            # Show the first anomaly in preview
            first_fail = results[results['status'] != 'NORMAL'].iloc[0]
            self._show_anomaly_frame(int(first_fail['frame']))
        else:
            self.status_badge.configure(text="PASSED: VALID INTEGRITY", fg_color=D_GREEN, text_color="black")
            self._display_first_frame()

        self.status_text.configure(text="Validation Complete")
        self.csv_btn.configure(state="normal")
        self.pdf_btn.configure(state="normal")
        self.folder_btn.configure(state="normal")
        self.upload_btn.configure(state="normal")
        self.analyze_btn.configure(state="normal")

    def _show_anomaly_frame(self, frame_idx):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        ret, frame = cap.read()
        if ret:
            self._update_preview(frame, is_anomaly=True)
        cap.release()

    def _download_csv(self):
        if self.csv_path:
            save = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="frame_audit.csv")
            if save: shutil.copy(self.csv_path, save)

    def _download_pdf(self):
        if self.pdf_path:
            save = filedialog.asksaveasfilename(defaultextension=".pdf", initialfile="Integrity_Report.pdf")
            if save: shutil.copy(self.pdf_path, save)

    def _open_results(self):
        if self.output_dir: os.startfile(self.output_dir)

    def _reset_ui(self):
        self.analyze_btn.configure(state="normal")
        self.upload_btn.configure(state="normal")
        self.status_text.configure(text="Ready")
        self.progress_bar.set(0)

if __name__ == "__main__":
    app = VideoAnalyzerApp()
    app.mainloop()
