import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import threading
import shutil
from analyzer.video_analyzer import VideoAnalyzer
from reports.report_generator import ReportGenerator

# Set Light Theme as requested by the mockup
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

class VideoAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Video Temporal Error Detector")
        self.geometry("1400x850")
        
        self.video_path = None
        self.output_dir = None
        self.csv_path = None
        self.txt_path = None
        self.analyzer = None
        self.output_base = os.path.join(os.getcwd(), "results")
        self.report_gen = ReportGenerator(self.output_base)

        self._setup_ui()

    def _setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # 1. Top Bar
        self.top_bar = ctk.CTkFrame(self, fg_color="#f8f9fa", border_width=1, border_color="#dee2e6")
        self.top_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.top_bar.grid_columnconfigure(1, weight=1) # Filename takes flexible space

        self.upload_btn = ctk.CTkButton(self.top_bar, text="Select Video", width=160, command=self._upload_video)
        self.upload_btn.grid(row=0, column=0, padx=10, pady=10)
        
        self.file_label = ctk.CTkLabel(self.top_bar, text="No video selected", anchor="w", fg_color="white", corner_radius=5)
        self.file_label.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.analyze_btn = ctk.CTkButton(self.top_bar, text="Run Analysis", width=160, 
                                         command=self._start_analysis, state="disabled",
                                         fg_color="#2ecc71", hover_color="#27ae60")
        self.analyze_btn.grid(row=0, column=2, padx=10, pady=10)

        # 2. Main content (Side-by-side)
        self.main_container = ctk.CTkFrame(self, fg_color="#f0f0f0")
        self.main_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.main_container.grid_columnconfigure(0, weight=6) # Table space
        self.main_container.grid_columnconfigure(1, weight=4) # Report space
        self.main_container.grid_rowconfigure(0, weight=1)

        # Left: Frame Data Preview
        self.preview_frame = ctk.CTkFrame(self.main_container, fg_color="white", border_width=1, border_color="#ccc")
        self.preview_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.preview_box = ctk.CTkTextbox(self.preview_frame, font=ctk.CTkFont(family="Courier", size=12), text_color="black", fg_color="white")
        self.preview_box.pack(fill="both", expand=True, padx=2, pady=2)
        self.preview_box.insert("0.0", "Frame analysis data will be displayed here...")
        self.preview_box.configure(state="disabled")

        # Right: Detailed Report
        self.report_frame = ctk.CTkFrame(self.main_container, fg_color="white", border_width=1, border_color="#ccc")
        self.report_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.report_box = ctk.CTkTextbox(self.report_frame, font=ctk.CTkFont(size=13), text_color="black", fg_color="white")
        self.report_box.pack(fill="both", expand=True, padx=5, pady=5)
        self.report_box.insert("0.0", "Detailed Report\n\nWaiting for analysis...")
        self.report_box.configure(state="disabled")

        # 3. Bottom Bar (Actions)
        self.bottom_bar = ctk.CTkFrame(self, fg_color="transparent")
        self.bottom_bar.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        
        self.export_csv_btn = ctk.CTkButton(self.bottom_bar, text="Export CSV", command=self._download_csv, state="disabled")
        self.export_csv_btn.pack(side="left", padx=5)
        
        self.export_pdf_btn = ctk.CTkButton(self.bottom_bar, text="Export PDF Report", command=self._download_pdf, state="disabled")
        self.export_pdf_btn.pack(side="left", padx=5)
        
        self.save_all_btn = ctk.CTkButton(self.bottom_bar, text="Open Results Folder", command=self._open_results, state="disabled")
        self.save_all_btn.pack(side="left", padx=5)

        self.zip_btn = ctk.CTkButton(self.bottom_bar, text="Export All (ZIP)", command=self._export_all_zip, state="disabled", fg_color="#f39c12", hover_color="#e67e22")
        self.zip_btn.pack(side="left", padx=5)
        
        
        
        # Status & Progress Footer
        self.status_bar = ctk.CTkFrame(self, height=35)
        self.status_bar.grid(row=3, column=0, sticky="ew", padx=0, pady=0)
        
        self.status_label = ctk.CTkLabel(self.status_bar, text="Status: Ready", text_color="#3498db")
        self.status_label.pack(side="left", padx=20)
        
        self.progress_bar = ctk.CTkProgressBar(self.status_bar, width=1100)
        self.progress_bar.pack(side="right", padx=10, pady=10)
        self.progress_bar.set(0)

    def _upload_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")])
        if path:
            self.video_path = path
            filename = os.path.basename(path)
            # Truncate if too long (e.g., > 50 chars) for display stability
            display_name = (filename[:47] + '...') if len(filename) > 50 else filename
            self.file_label.configure(text=display_name, text_color="black")
            self.analyze_btn.configure(state="normal")
            self.status_label.configure(text="Status: File Loaded")

    def _start_analysis(self):
        if not self.video_path:
            return
        
        self.analyze_btn.configure(state="disabled")
        self.upload_btn.configure(state="disabled")
        self.status_label.configure(text="Status: Analyzing...")
        
        threading.Thread(target=self._run_analysis_thread, daemon=True).start()

    def _run_analysis_thread(self):
        try:
            self.analyzer = VideoAnalyzer(self.video_path)
            
            def update_progress(current, total):
                self.progress_bar.set(current / total)
            
            self.analyzer.collect_metrics(progress_callback=update_progress)
            results_df, summary = self.analyzer.classify_frames()
            
            self.output_dir, self.csv_path, self.txt_path, summary_text = self.report_gen.generate_reports(
                results_df, summary, self.video_path,
                custom_name=None,
                analyzer_metadata=self.analyzer.processing_metadata
            )
            
            self.after(0, lambda: self._update_ui_results(summary, summary_text, results_df))
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            self.after(0, lambda: self._reset_ui())

    def _update_ui_results(self, summary, summary_text, results_df):
        self.preview_box.configure(state="normal")
        self.preview_box.delete("0.0", "end")
        
        # Filter for GUI Display as requested: Frame | Timestamp | Time Diff | Motion | SSIM | Status
        display_df = results_df[['frame', 'timestamp', 'time_diff', 'motion_score', 'ssim', 'status']].copy()
        display_df.columns = ['Frame', 'Timestamp', 'Time Diff', 'Motion', 'SSIM', 'Status']
        
        # Format for readability
        display_df['Timestamp'] = display_df['Timestamp'].map('{:,.3f}'.format)
        display_df['Time Diff'] = display_df['Time Diff'].map('{:,.4f}'.format)
        display_df['Motion'] = display_df['Motion'].map('{:,.3f}'.format)
        display_df['SSIM'] = display_df['SSIM'].map('{:,.4f}'.format)
        
        preview_text = display_df.to_string(index=False, max_rows=2000, justify='left')
        self.preview_box.insert("0.0", preview_text)
        self.preview_box.configure(state="disabled")

        self.report_box.configure(state="normal")
        self.report_box.delete("0.0", "end")
        self.report_box.insert("0.0", "Detailed Report\n" + "="*30 + "\n" + summary_text)
        self.report_box.configure(state="disabled")

        self.status_label.configure(text="Status: Complete", text_color="#2ecc71")
        
        # Enable all buttons
        btns = [self.export_csv_btn, self.export_pdf_btn, self.save_all_btn, self.zip_btn]
        for btn in btns:
            btn.configure(state="normal")
        
        self.analyze_btn.configure(state="normal")
        self.upload_btn.configure(state="normal")

    def _copy_summary(self):
        self.clipboard_clear()
        self.clipboard_append(self.report_box.get("0.0", "end"))
        messagebox.showinfo("Clipboard", "Summary report copied to clipboard!")

    def _download_csv(self):
        if self.csv_path and os.path.exists(self.csv_path):
            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                initialfile="frame_classification.csv",
                filetypes=[("CSV files", "*.csv")]
            )
            if save_path:
                shutil.copy(self.csv_path, save_path)
                messagebox.showinfo("Success", f"CSV saved to: {save_path}")

    def _download_pdf(self):
        pdf_path = os.path.join(self.output_dir, "analysis_report.pdf")
        if os.path.exists(pdf_path):
            save_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                initialfile="analysis_report.pdf",
                filetypes=[("PDF files", "*.pdf")]
            )
            if save_path:
                shutil.copy(pdf_path, save_path)
                messagebox.showinfo("Success", f"PDF saved to: {save_path}")

    def _export_all_zip(self):
        if self.output_dir and os.path.exists(self.output_dir):
            save_path = filedialog.asksaveasfilename(
                defaultextension=".zip",
                initialfile=f"{os.path.basename(self.output_dir)}.zip",
                filetypes=[("ZIP files", "*.zip")]
            )
            if save_path:
                # Remove extension if user added it, as make_archive adds its own
                if save_path.lower().endswith('.zip'):
                    save_path = save_path[:-4]
                shutil.make_archive(save_path, 'zip', self.output_dir)
                messagebox.showinfo("Success", f"All results zipped to: {save_path}.zip")

    def _open_results(self):
        if self.output_dir and os.path.exists(self.output_dir):
            os.startfile(self.output_dir)

    def _reset_ui(self):
        self.analyze_btn.configure(state="normal")
        self.upload_btn.configure(state="normal")
        self.status_label.configure(text="Status: Ready", text_color="#3498db")
        self.progress_bar.set(0)

if __name__ == "__main__":
    app = VideoAnalyzerApp()
    app.mainloop()
