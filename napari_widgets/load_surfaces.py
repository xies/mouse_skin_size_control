"""
Napari mesh time-series loader
===============================

Loads a folder of mesh .npz files (each containing `vertices`, `faces`,
`values`) laid out like:

    root/
        0. Day 0/mesh.npz
        1. Day 1/mesh.npz
        2. Day 4/mesh.npz
        ...

Parses the timestamp from each parent folder name using its leading index
number (e.g. "0. Day 0" -> 0, "1. Day 1" -> 1), sorts t0 -> tN, and
concatenates every mesh into a single 4D napari Surface layer (time is the
extra leading dimension, so you get a slider to scrub through timepoints).

HOW TO USE
----------
Run this script: `python napari_mesh_timeseries.py`, click "Select Root
Folder", then "Load & Concatenate".
"""

import re
import traceback
from pathlib import Path

import numpy as np
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

LEADING_NUM_PATTERN = re.compile(r"^\s*(\d+)")


def parse_timestamp(folder_name: str):
    """
    Returns an integer timestamp for a folder name like '0. Day 0' or
    '1. Day 3', using the leading index number. Returns None if the folder
    name doesn't start with a number.
    """
    m = LEADING_NUM_PATTERN.match(folder_name)
    if m:
        return int(m.group(1))
    return None


class MeshTimeSeriesWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.root_dir = None
        self.entries = []  # list of (timestamp, npz_path)

        layout = QVBoxLayout()

        # --- select root folder ---
        root_row = QHBoxLayout()
        self.root_btn = QPushButton("Select Root Folder")
        self.root_btn.clicked.connect(self.select_root)
        self.root_label = QLabel("No folder selected")
        root_row.addWidget(self.root_btn)
        root_row.addWidget(self.root_label)
        layout.addLayout(root_row)

        # --- load & concatenate ---
        self.load_btn = QPushButton("Load && Concatenate")
        self.load_btn.clicked.connect(self.load_and_concatenate)
        self.load_btn.setEnabled(False)
        layout.addWidget(self.load_btn)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.setLayout(layout)

    def log_msg(self, msg: str):
        self.log.appendPlainText(msg)
        QApplication.processEvents()

    def select_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select root folder")
        if not path:
            return
        self.root_dir = Path(path)
        self.root_label.setText(str(self.root_dir))
        self.log.clear()
        self._scan()

    def _scan(self):
        """Find one .npz per immediate subfolder and parse its timestamp."""
        candidates = []
        for sub in sorted(self.root_dir.iterdir()):
            if not sub.is_dir():
                continue
            mesh_path = sub / "mesh.npz"
            if not mesh_path.exists():
                continue
            ts = parse_timestamp(sub.name)
            candidates.append((ts, sub.name, mesh_path))

        if not candidates:
            self.log_msg("No 'mesh.npz' found in any subfolder.")
            self.entries = []
            self.load_btn.setEnabled(False)
            return

        missing = [name for ts, name, _ in candidates if ts is None]
        if missing:
            self.log_msg(
                f"WARNING: couldn't parse a timestamp for: {missing} "
                f"-> falling back to alphabetical order for those."
            )
            # fall back: alphabetical rank as timestamp, offset past real ones
            max_ts = max((ts for ts, _, _ in candidates if ts is not None), default=-1)
            fallback = max_ts + 1
            fixed = []
            for ts, name, p in candidates:
                if ts is None:
                    fixed.append((fallback, name, p))
                    fallback += 1
                else:
                    fixed.append((ts, name, p))
            candidates = fixed

        candidates.sort(key=lambda x: x[0])
        self.entries = [(ts, p) for ts, name, p in candidates]

        self.log_msg(f"Found {len(self.entries)} mesh(es), in order:")
        for ts, p in self.entries:
            self.log_msg(f"  t={ts:<5} {p}")

        self.load_btn.setEnabled(True)

    def load_and_concatenate(self):
        n_total = len(self.entries)
        self.progress.setMinimum(0)
        self.progress.setMaximum(n_total)
        self.progress.setValue(0)

        all_vertices = []
        all_faces = []
        all_values = []
        vertex_offset = 0

        for i, (ts, path) in enumerate(self.entries):
            try:
                data = np.load(path)
                vertices = data["vertices"]  # (N, 3)
                faces = data["faces"]        # (M, 3)
                values = data["values"] if "values" in data else np.zeros(len(vertices))

                # prepend timestamp as leading coordinate -> napari 4D surface
                t_col = np.full((vertices.shape[0], 1), ts, dtype=vertices.dtype)
                vertices_4d = np.hstack([t_col, vertices])

                all_vertices.append(vertices_4d)
                all_faces.append(faces + vertex_offset)
                all_values.append(values)

                vertex_offset += vertices.shape[0]
                self.log_msg(f"Loaded t={ts} ({path.parent.name}): {vertices.shape[0]} verts")
            except Exception:
                self.log_msg(f"ERROR loading {path}:\n{traceback.format_exc()}")

            self.progress.setValue(i + 1)

        if not all_vertices:
            self.log_msg("Nothing loaded, aborting.")
            return

        vertices_cat = np.vstack(all_vertices)
        faces_cat = np.vstack(all_faces)
        values_cat = np.concatenate(all_values)

        self.viewer.add_surface(
            (vertices_cat, faces_cat, values_cat),
            name=f"{self.root_dir.name}_timeseries",
            colormap='twilight',
            contrast_limits=[-.5,.5]
        )
        self.log_msg("Done. Added concatenated surface layer.")


def main():
    viewer = napari.Viewer()
    widget = MeshTimeSeriesWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Mesh Time Series Loader", area="right")
    napari.run()


if __name__ == "__main__":
    main()
