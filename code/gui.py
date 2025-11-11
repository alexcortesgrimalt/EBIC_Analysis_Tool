import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from .sem_viewer import SEMViewer
from .processor import SEMTiffProcessorWrapper

class SEMStarterGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SEM TIFF Processor")

        self.processor = SEMTiffProcessorWrapper()
        self.tiff_dir = None
        self.tiff_files = []

        self.build_gui()
        self.root.mainloop()

    def build_gui(self):
        tk.Label(self.root, text="Select TIFF Folder:").pack(pady=5)
        self.folder_label = tk.Label(self.root, text="No folder selected", fg="blue")
        self.folder_label.pack(pady=5)

        tk.Button(self.root, text="Browse Folder", command=self.browse_folder).pack(pady=5)

        self.file_listbox = tk.Listbox(self.root, selectmode=tk.MULTIPLE, width=50)
        self.file_listbox.pack(pady=5)

        tk.Button(self.root, text="Process Selected Files", command=self.process_files).pack(pady=10)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.tiff_dir = Path(folder)
            self.folder_label.config(text=str(self.tiff_dir))
            self.update_file_list()

    def update_file_list(self):
        self.file_listbox.delete(0, tk.END)
        self.tiff_files = [f for f in self.tiff_dir.iterdir() if f.suffix.lower() in [".tif", ".tiff"]]
        self.tiff_files.sort()

        if not self.tiff_files:
            messagebox.showwarning("No Files", "No TIFF files found in the selected folder.")
            return

        for f in self.tiff_files:
            self.file_listbox.insert(tk.END, f.name)

    def process_files(self):
        selected_indices = self.file_listbox.curselection()
        selected_files = [self.tiff_files[i] for i in selected_indices] if selected_indices else self.tiff_files

        if not selected_files:
            messagebox.showwarning("No Selection", "No files selected for processing.")
            return

        output_folder = Path(filedialog.askdirectory(title="Select Output Folder") or "tiff_test_output")
        processed_names = self.processor.process_files(selected_files, output_folder)

        # If multiple files were selected, ask whether they're a voltage sweep of the same sample.
        if len(selected_files) > 1:
            is_sweep = messagebox.askyesno("Voltage sweep?", "Open selected files as a voltage sweep of the same sample (align across files)?")
        else:
            is_sweep = False

        if not processed_names:
            messagebox.showerror("Error", "No processed frames found.")
            return

        # Dataset selection / sweep handling
        if is_sweep and len(processed_names) > 1:
            # Load all processed datasets in background and open a multi-dataset viewer
            all_maps = self.processor.load_multiple_maps(output_folder, processed_names)

            # Ask which dataset to display initially
            sample_name = self.choose_dataset(processed_names)
            if sample_name is None:
                return

            # find index
            start_index = processed_names.index(sample_name)

            # open SEMViewer with sweep datasets
            if messagebox.askyesno("Open Viewer", f"Open interactive viewer for sweep (showing {sample_name})?"):
                # transform all_maps into sweep_datasets list of dicts expected by SEMViewer
                sweep_datasets = []
                for pix, cur, psize, sname, framesz in all_maps:
                    sweep_datasets.append({
                        'pixel_maps': pix,
                        'current_maps': cur,
                        'pixel_size': psize,
                        'sample_name': sname,
                        'frame_sizes': framesz
                    })

                viewer = SEMViewer(sweep_datasets[start_index]['pixel_maps'],
                                   sweep_datasets[start_index]['current_maps'],
                                   sweep_datasets[start_index]['pixel_size'],
                                   sweep_datasets[start_index]['sample_name'],
                                   frame_sizes=sweep_datasets[start_index]['frame_sizes'],
                                   sweep_datasets=sweep_datasets,
                                   sweep_start_index=start_index)
                viewer.show()
            else:
                messagebox.showinfo("Done", f"Processed {len(processed_names)} datasets for sweep. Viewer skipped.")
        else:
            # Dataset selection
            if len(processed_names) == 1:
                self.open_viewer(output_folder, processed_names[0])
            else:
                sample_name = self.choose_dataset(processed_names)
                if sample_name:
                    self.open_viewer(output_folder, sample_name)

    def choose_dataset(self, dataset_names):
        win = tk.Toplevel(self.root)
        win.title("Choose Dataset")
        selection = []

        tk.Label(win, text="Select a processed dataset to view:").pack(pady=5)
        listbox = tk.Listbox(win, selectmode=tk.SINGLE, width=50)
        listbox.pack(pady=5)

        for name in dataset_names:
            listbox.insert(tk.END, name)

        def on_select():
            sel = listbox.curselection()
            if sel:
                selection.append(dataset_names[sel[0]])
                win.destroy()
            else:
                tk.messagebox.showwarning("No Selection", "Please choose a dataset.")

        tk.Button(win, text="Open Selected", command=on_select).pack(pady=10)
        win.wait_window()
        return selection[0] if selection else None

    def open_viewer(self, output_folder: Path, sample_name: str):
        pixel_maps, current_maps, pixel_size, sample_name, frame_sizes = self.processor.load_maps(output_folder, sample_name)

        if not frame_sizes:
            messagebox.showerror("Error", f"No processed frames found for {sample_name}.")
            return

        if messagebox.askyesno("Open Viewer", f"Open interactive viewer for {sample_name}?"):
            viewer = SEMViewer(pixel_maps, current_maps, pixel_size, sample_name, frame_sizes)
            viewer.show()
        else:
            messagebox.showinfo("Done", f"Processed dataset '{sample_name}' ready. Viewer skipped.")
