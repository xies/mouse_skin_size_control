"""
Napari batch mesh-extraction tool
==================================

Loads a single .TIF file (either TZYX or ZYX) and runs your mesh-extraction
function on every volume (every timepoint if TZYX, or the single volume if
ZYX), saving one mesh output per volume.

HOW TO USE
----------
1. Scroll down to the `process_volume()` function below.
2. Replace the placeholder line with your actual function call.
3. Run this script: `python napari_mesh_batch.py`

That's it — everything else (file loading, dimension detection, looping,
progress bar, logging, saving) is already wired up.
"""

import traceback
from pathlib import Path

import numpy as np
import tifffile
import napari
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QProgressBar,
    QPlainTextEdit,
    QFileDialog,
    QApplication,
)


# ============================================================================
# >>> EDIT HERE: plug in your mesh-extraction function <<<
# ============================================================================
def process_volume(volume: np.ndarray, output_path: Path):
    """
    Runs on a single 3D (ZYX) volume and should extract + save a mesh.

    `volume`      : np.ndarray, shape (Z, Y, X)
    `output_path` : Path where you should save the mesh (extension-free stem
                    provided below, e.g. add '.obj' / '.ply' yourself)

    Replace the two lines below with your own function call, e.g.:

        mesh = my_mesh_extraction_function(volume, some_param=123)
        mesh.export(str(output_path.with_suffix(".obj")))

    or if your function returns (verts, faces, normals, values) like
    skimage.measure.marching_cubes, save them however you normally do.
    """

    # ---- PLACEHOLDER (delete once you paste in your real code) ----
    print(f"[placeholder] would process volume of shape {volume.shape} "
          f"-> {output_path}")

    # -----------------------------------------------------------------


# ============================================================================
# GUI — nothing below this line needs to be edited
# ============================================================================
class BatchMeshWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.filepath = None
        self.volume = None          # loaded array, TZYX or ZYX
        self.is_time_series = False
        self.output_dir = None

        layout = QVBoxLayout()

        # --- load file ---
        load_row = QHBoxLayout()
        self.load_btn = QPushButton("Load .TIF")
        self.load_btn.clicked.connect(self.load_tif)
        self.file_label = QLabel("No file loaded")
        load_row.addWidget(self.load_btn)
        load_row.addWidget(self.file_label)
        layout.addLayout(load_row)

        # --- output folder ---
        out_row = QHBoxLayout()
        self.out_btn = QPushButton("Select Output Folder")
        self.out_btn.clicked.connect(self.select_output_dir)
        self.out_label = QLabel("No output folder selected")
        out_row.addWidget(self.out_btn)
        out_row.addWidget(self.out_label)
        layout.addLayout(out_row)

        # --- run ---
        self.run_btn = QPushButton("Run Batch")
        self.run_btn.clicked.connect(self.run_batch)
        self.run_btn.setEnabled(False)
        layout.addWidget(self.run_btn)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.setLayout(layout)

    def log_msg(self, msg: str):
        self.log.appendPlainText(msg)
        QApplication.processEvents()

    def load_tif(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select TIF file", "", "TIFF files (*.tif *.tiff)"
        )
        if not path:
            return

        self.filepath = Path(path)
        try:
            self.volume = tifffile.imread(str(self.filepath))
        except Exception as e:
            self.log_msg(f"ERROR loading file: {e}")
            return

        ndim = self.volume.ndim
        if ndim == 4:
            self.is_time_series = True
            n_t = self.volume.shape[0]
            self.file_label.setText(
                f"{self.filepath.name}  (TZYX, T={n_t}, shape={self.volume.shape})"
            )
        elif ndim == 3:
            self.is_time_series = False
            self.file_label.setText(
                f"{self.filepath.name}  (ZYX, shape={self.volume.shape})"
            )
        else:
            self.log_msg(
                f"ERROR: expected 3D (ZYX) or 4D (TZYX) array, got shape {self.volume.shape}"
            )
            self.volume = None
            return

        self.viewer.add_image(self.volume, name=self.filepath.name)
        self.log_msg(f"Loaded {self.filepath.name}, shape {self.volume.shape}")
        self._maybe_enable_run()

    def select_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not path:
            return
        self.output_dir = Path(path)
        self.out_label.setText(str(self.output_dir))
        self._maybe_enable_run()

    def _maybe_enable_run(self):
        self.run_btn.setEnabled(self.volume is not None and self.output_dir is not None)

    def run_batch(self):
        stem = self.filepath.stem
        n_total = self.volume.shape[0] if self.is_time_series else 1
        self.progress.setMinimum(0)
        self.progress.setMaximum(n_total)
        self.progress.setValue(0)

        for i in range(n_total):
            vol = self.volume[i] if self.is_time_series else self.volume
            out_name = f"{stem}_t{i:03d}" if self.is_time_series else stem
            out_path = self.output_dir / out_name

            try:
                self.log_msg(f"Processing {out_name} ...")
                process_volume(vol, out_path)
                self.log_msg(f"  done -> {out_path}")
            except Exception:
                self.log_msg(f"  ERROR on {out_name}:\n{traceback.format_exc()}")

            self.progress.setValue(i + 1)

        self.log_msg("Batch complete.")


def main():
    viewer = napari.Viewer()
    widget = BatchMeshWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Batch Mesh Extraction", area="right")
    napari.run()


if __name__ == "__main__":
    main()
