import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import threading
import shutil
import cv2
from PIL import Image, ImageTk
from analyzer.video_analyzer import VideoAnalyzer
from reports.report_generator import ReportGenerator

# Unified Forensic Branding
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
        self.geometry("1450x920")
        self.configure(fg_color=D_BG)
        
        # Absolute Forensic State
        self.video_path = None
        self.output_dir = None
        self.csv_path = None
        self.pdf_path = None
        self.log_path = None
        
        self.analyzer = None
        self.report_gen = ReportGenerator("results")
        self.current_frame_img = None

        self._setup_ui()

    def _setup_ui(self):
        self.grid_columnconfigure(1, weight=10) 
        self.grid_columnconfigure(2, weight=3) 
        self.grid_rowconfigure(0, weight=1)

        # 1. Sidebar Panel
        self.sidebar = ctk.CTkFrame(self, fg_color=D_CARD, corner_radius=0, width=280)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        ctk.CTkLabel(self.sidebar, text="VTED", font=ctk.CTkFont(size=28, weight="bold"), text_color=D_ACCENT).pack(pady=(30, 0))
        ctk.CTkLabel(self.sidebar, text="Forensic Engine v2.2", font=ctk.CTkFont(size=12), text_color=D_TEXT).pack(pady=(0, 40))

        # Main Navigation
        self.upload_btn = self._create_sidebar_btn("Upload Video", self._upload_video, color=D_ACCENT)
        self.analyze_btn = self._create_sidebar_btn("Run Analysis", self._start_analysis, state="disabled", color="#1C2128")
        
        # Export Category
        ctk.CTkLabel(self.sidebar, text="REPORTING & EXPORTS", font=ctk.CTkFont(size=11, weight="bold"), text_color="#586069").pack(pady=(40, 10))
        
        self.csv_btn = self._create_sidebar_btn("Download CSV", self._download_csv, state="disabled")
        self.pdf_btn = self._create_sidebar_btn("Download PDF", self._download_pdf, state="disabled")
        self.summary_btn = self._create_sidebar_btn("Forensic Summary", self._download_summary, state="disabled")
        self.zip_btn = self._create_sidebar_btn("Export All (ZIP)", self._download_zip, state="disabled")
        self.folder_btn = self._create_sidebar_btn("Open Results Folder", self._open_results, state="disabled")

        # 2. Main Interface (Center)
        self.main_view = ctk.CTkFrame(self, fg_color=D_BG, corner_radius=0)
        self.main_view.grid(row=0, column=1, sticky="nsew", padx=25, pady=25)
        
        self.video_display = ctk.CTkLabel(self.main_view, text="NO DATA LOADED", fg_color=D_CARD, corner_radius=15, 
                                          font=ctk.CTkFont(size=16), text_color=D_TEXT)
        self.video_display.pack(fill="both", expand=True, padx=10, pady=10)

        self.status_badge = ctk.CTkLabel(self.main_view, text="SYSTEM STANDBY", fg_color=D_CARD, corner_radius=30,
                                         height=60, width=400, font=ctk.CTkFont(size=22, weight="bold"), text_color=D_TEXT)
        self.status_badge.pack(pady=15)

        self.desc_box = ctk.CTkTextbox(self.main_view, height=200, fg_color=D_CARD, text_color=D_TEXT, font=ctk.CTkFont(size=13), border_width=1, border_color="#30363D")
        self.desc_box.pack(fill="x", padx=10, pady=(0, 10))
        self.desc_box.insert("0.0", "VTED v2.2 | Forensic Temporal Validator\nStep 1: Upload a broadcast video segment.")

        # 3. Summary Panel (Right)
        self.summary_panel = ctk.CTkFrame(self, fg_color=D_CARD, corner_radius=15)
        self.summary_panel.grid(row=0, column=2, sticky="nsew", padx=(0, 25), pady=25)
        
        ctk.CTkLabel(self.summary_panel, text="Audit Summary", font=ctk.CTkFont(size=20, weight="bold"), text_color=D_ACCENT).pack(pady=25)

        self.stats_displays = {}
        for key in ["Video Name", "Total Frames", "Duration", "Metadata FPS", "Computed FPS", "Drop Count", "Merge Count", "Integrity Score"]:
            container = ctk.CTkFrame(self.summary_panel, fg_color="transparent")
            container.pack(fill="x", padx=25, pady=8)
            ctk.CTkLabel(container, text=key, text_color=D_TEXT, font=ctk.CTkFont(size=13)).pack(side="left")
            val_label = ctk.CTkLabel(container, text="--", text_color="white", font=ctk.CTkFont(size=13, weight="bold"))
            val_label.pack(side="right")
            self.stats_displays[key] = val_label

        self.progress_bar = ctk.CTkProgressBar(self.summary_panel, width=220, fg_color="#30363D", progress_color=D_ACCENT)
        self.progress_bar.pack(pady=(50, 10))
        self.progress_bar.set(0)
        self.status_text = ctk.CTkLabel(self.summary_panel, text="Awaiting Input", font=ctk.CTkFont(size=12), text_color=D_TEXT)
        self.status_text.pack()

    def _create_sidebar_btn(self, text, command, state="normal", color=None):
        btn = ctk.CTkButton(self.sidebar, 
                             text=text, 
                             command=command, 
                             state=state,
                             fg_color=color if color else "#21262D", 
                             text_color="white",
                             border_width=1,
                             border_color=D_ACCENT if state == "normal" else "#30363D",
                             hover_color="#30363D",
                             height=45, 
                             font=ctk.CTkFont(size=13, weight="bold"))
        btn.pack(fill="x", padx=25, pady=8)
        return btn

    def _upload_video(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")])
        if path:
            self.video_path = os.path.abspath(path)
            self.analyze_btn.configure(state="normal", border_color=D_ACCENT, fg_color="#21262D")
            fname = os.path.basename(path)
            self.stats_displays["Video Name"].configure(text=fname[:20] + ".." if len(fname) > 20 else fname)
            self.status_text.configure(text="Video Loaded Successfully")
            self._display_snapshot()

    def _display_snapshot(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if ret: self._update_preview(frame)
        cap.release()

    def _update_preview(self, frame, is_anomaly=False):
        h, w = frame.shape[:2]
        display_w = 880
        display_h = int(h * (display_w / w))
        if is_anomaly:
            red = np.zeros_like(frame)
            red[:, :] = (0, 0, 180)
            frame = cv2.addWeighted(frame, 0.7, red, 0.3, 0)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((display_w, display_h), Image.Resampling.LANCZOS)
        self.current_frame_img = ImageTk.PhotoImage(img)
        self.video_display.configure(image=self.current_frame_img, text="")

    def _start_analysis(self):
        self._toggle_buttons("disabled")
        self.status_text.configure(text="Analyzing Frames...")
        self.desc_box.delete("0.0", "end")
        self.desc_box.insert("0.0", "Processing forensic metrics...")
        threading.Thread(target=self._run_audit_pipeline, daemon=True).start()

    def _run_audit_pipeline(self):
        try:
            self.analyzer = VideoAnalyzer(self.video_path)
            def tick(c, t): self.progress_bar.set(c / t)
            self.analyzer.collect_metrics(progress_callback=tick)
            results, summary = self.analyzer.classify_frames()
            
            o_dir, c_path, p_path = self.report_gen.generate_reports(results, summary, self.video_path)
            l_path = os.path.join(o_dir, "analysis_log.txt")
            
            self.after(0, lambda: self._apply_results_to_ui(summary, results, o_dir, c_path, p_path, l_path))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Forensic Error", f"Analysis Pipeline Failed:\n{str(e)}"))
            self.after(0, self._reset_ui)

    def _apply_results_to_ui(self, summ, res, o_dir, c_path, p_path, l_path):
        self.output_dir = os.path.abspath(o_dir)
        self.csv_path = os.path.abspath(c_path)
        self.pdf_path = os.path.abspath(p_path)
        self.log_path = os.path.abspath(l_path)

        # Refresh Labels
        self.stats_displays["Total Frames"].configure(text=str(summ['total_frames']))
        self.stats_displays["Duration"].configure(text=f"{summ['duration']:.2f}s")
        self.stats_displays["Metadata FPS"].configure(text=f"{summ['metadata_fps']:.2f}")
        self.stats_displays["Computed FPS"].configure(text=f"{summ['computed_fps']:.2f}")
        self.stats_displays["Drop Count"].configure(text=str(summ['drop_count']), text_color=D_RED if summ['drop_count'] > 0 else "white")
        self.stats_displays["Merge Count"].configure(text=str(summ['merge_count']), text_color=D_GREEN if summ['merge_count'] > 0 else "white")
        self.stats_displays["Integrity Score"].configure(text=f"{summ['integrity_score']:.2f}%")

        self.desc_box.delete("0.0", "end")
        self.desc_box.insert("0.0", summ.get('report_description'))

        if summ['drop_count'] > 0 or summ['merge_count'] > 0:
            self.status_badge.configure(text="ANOMALIES CONFIRMED", fg_color=D_RED, text_color="white")
            self._jump_to_fail(res)
        else:
            self.status_badge.configure(text="INTEGRITY VALIDATED", fg_color=D_GREEN, text_color="black")
            self._display_snapshot()

        self.status_text.configure(text="Audit Success")
        self._toggle_buttons("normal")

    def _toggle_buttons(self, state):
        btns = [self.csv_btn, self.pdf_btn, self.summary_btn, self.zip_btn, self.folder_btn, self.upload_btn, self.analyze_btn]
        for b in btns:
            b.configure(state=state, border_color=D_ACCENT if state == "normal" else "#30363D")

    def _jump_to_fail(self, results):
        try:
            frame_idx = int(results[results['status'] != 'NORMAL'].iloc[0]['frame'])
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            ret, frame = cap.read()
            if ret: self._update_preview(frame, is_anomaly=True)
            cap.release()
        except: pass

    def _download_csv(self):
        if not self.csv_path: return
        file_path = filedialog.asksaveasfilename(parent=self, defaultextension=".csv", initialfile="frame_audit.csv", title="Save Forensic CSV")
        if file_path:
            shutil.copy2(self.csv_path, file_path)
            messagebox.showinfo("Forensic Data", f"CSV Saved:\n{file_path}")

    def _download_pdf(self):
        if not self.pdf_path: return
        file_path = filedialog.asksaveasfilename(parent=self, defaultextension=".pdf", initialfile="Integrity_Report.pdf", title="Save Forensic PDF")
        if file_path:
            shutil.copy2(self.pdf_path, file_path)
            messagebox.showinfo("Forensic Report", f"PDF Saved:\n{file_path}")

    def _download_summary(self):
        if not self.log_path: return
        file_path = filedialog.asksaveasfilename(parent=self, defaultextension=".txt", initialfile="Analysis_Log.txt", title="Save Forensic Log")
        if file_path:
            shutil.copy2(self.log_path, file_path)
            messagebox.showinfo("Forensic Log", f"Log Saved:\n{file_path}")

    def _download_zip(self):
        if not self.output_dir: return
        file_path = filedialog.asksaveasfilename(parent=self, defaultextension=".zip", initialfile="VTED_Results.zip", title="Save Results ZIP")
        if file_path:
            base = os.path.splitext(file_path)[0]
            shutil.make_archive(base, 'zip', self.output_dir)
            messagebox.showinfo("Forensic Archive", f"Session archived to:\n{base}.zip")

    def _open_results(self):
        if self.output_dir: os.startfile(self.output_dir)

    def _reset_ui(self):
        self._toggle_buttons("normal")
        self.status_text.configure(text="Ready")
        self.progress_bar.set(0)

if __name__ == "__main__":
    app = VideoAnalyzerApp()
    app.mainloop()
