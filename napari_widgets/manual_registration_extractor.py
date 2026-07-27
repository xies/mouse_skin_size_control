"""
Napari plugin: Manual Rigid Registration Parameter Extractor
============================================================
Extracts rigid transformation parameters (translation, rotation, scale)
from napari image layers after the user has manually aligned them using
the layer transform controls (double-click a layer to activate transforms).

Usage
-----
1. Load your reference and moving images as separate Image layers in napari.
2. Double-click a layer to activate the transform handles; drag/rotate to align.
3. Open the plugin (Plugins > Rigid Registration Extractor).
4. Select the reference layer and one or more moving layers.
5. Click "Extract Parameters" to read out the transforms.
6. Click "Save CSV…" to open a file dialog and save results.

Dependencies
------------
    pip install napari numpy scipy

Works with napari >= 0.4.18 (affine transform stored on each layer).
Requires qtpy (bundled with napari).
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------

def decompose_affine_2d(matrix: np.ndarray) -> dict:
    """
    Decompose a 3x3 homogeneous affine matrix (napari 2D convention) into
    rigid + scale components.

    napari stores transforms as (ndim+1) x (ndim+1) homogeneous matrices
    acting on (z, y, x) or (y, x) coordinates.

    Returns
    -------
    dict with keys:
        translation_yx  : (ty, tx) in pixels
        rotation_deg    : rotation angle in degrees (CCW positive)
        scale_yx        : (sy, sx) scale factors
        shear           : shear parameter (non-rigid warning if != 0)
        matrix          : original 3x3 matrix
    """
    assert matrix.shape == (3, 3), "Expected 3x3 homogeneous matrix for 2D"

    # Translation is the last column (rows 0,1)
    ty, tx = matrix[0, 2], matrix[1, 2]

    # Extract the 2x2 linear part
    A = matrix[:2, :2]

    # Decompose via SVD: A = U @ diag(s) @ Vt
    # Rotation part: R = U @ Vt  (removes scale/shear)
    U, s, Vt = np.linalg.svd(A)

    # Scale factors are the singular values
    sy, sx = s[0], s[1]

    # Pure rotation matrix
    R = U @ Vt

    # Rotation angle (handle reflection: det should be +1 for proper rotation)
    det = np.linalg.det(R)
    if det < 0:
        # Reflection present — flip sign of last singular vector
        U[:, -1] *= -1
        s[-1]    *= -1
        R = U @ Vt

    angle_rad = np.arctan2(R[1, 0], R[0, 0])
    angle_deg = np.degrees(angle_rad)

    # Shear (residual after removing scale and rotation)
    # If the original matrix is purely rigid+scale, shear ~ 0
    R_scaled = np.diag(s) @ Vt  # what's left after U
    # shear estimate: off-diagonal of R_scaled normalised
    shear = float(R_scaled[0, 1] / s[0]) if s[0] > 0 else 0.0

    return {
        "translation_yx": (float(ty), float(tx)),
        "rotation_deg": float(angle_deg),
        "scale_yx": (float(sy), float(sx)),
        "shear": shear,
        "matrix": matrix.tolist(),
    }


def decompose_affine_3d(matrix: np.ndarray) -> dict:
    """
    Decompose a 4x4 homogeneous affine matrix (napari 3D convention) into
    rigid + scale components.

    Returns
    -------
    dict with keys:
        translation_zyx : (tz, ty, tx) in pixels/voxels
        rotation_euler_deg : (rx, ry, rz) intrinsic XYZ Euler angles in degrees
        scale_zyx       : (sz, sy, sx)
        rotation_matrix : 3x3 pure rotation matrix
        matrix          : original 4x4 matrix
    """
    assert matrix.shape == (4, 4), "Expected 4x4 homogeneous matrix for 3D"

    tz, ty, tx = matrix[0, 3], matrix[1, 3], matrix[2, 3]

    A = matrix[:3, :3]
    U, s, Vt = np.linalg.svd(A)

    sz, sy, sx = s[0], s[1], s[2]

    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        s[-1]    *= -1
        R = U @ Vt

    # Intrinsic XYZ Euler angles from rotation matrix
    # Using scipy for robustness (handles gimbal lock gracefully)
    try:
        from scipy.spatial.transform import Rotation
        euler_deg = Rotation.from_matrix(R).as_euler("xyz", degrees=True).tolist()
    except ImportError:
        # Fallback manual extraction (ZYX convention)
        sy_val = -R[2, 0]
        if abs(sy_val) < 1.0 - 1e-6:
            rx = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
            ry = float(np.degrees(np.arcsin(sy_val)))
            rz = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        else:  # Gimbal lock
            rx = float(np.degrees(np.arctan2(-R[1, 2], R[1, 1])))
            ry = float(np.degrees(np.arcsin(sy_val)))
            rz = 0.0
        euler_deg = [rx, ry, rz]

    return {
        "translation_zyx": (float(tz), float(ty), float(tx)),
        "rotation_euler_xyz_deg": euler_deg,
        "scale_zyx": (float(sz), float(sy), float(sx)),
        "rotation_matrix": R.tolist(),
        "matrix": matrix.tolist(),
    }


def _full_affine_matrix(layer) -> np.ndarray:
    """
    Return the full ndim+1 square data->world affine matrix for a layer,
    composed from public API properties only.

    napari's transform chain has two stages relevant to us:

      data2physical : encodes layer.scale and layer.translate
                      M_d2p = diag([*scale, 1]),  translation in last col
      physical2world: encodes layer.affine (GUI transform mode updates this)
                      M_affine = layer.affine.affine_matrix

    Full matrix:  M_full = M_affine @ M_d2p

    Why not use layer._transforms.simplified?
      That internal chain also includes a tile2data stage for multiscale
      layers (and other internals), which adds extra dimensions and is a
      private API.  Building from public properties is safer and clearer.

    Why not use layer.affine.affine_matrix alone?
      GUI transform mode (double-click drag) updates layer.affine ONLY.
      layer.scale and layer.translate remain unchanged (napari issue #6446).
      Reading layer.affine alone therefore misses any pre-existing voxel
      spacing or translate, corrupting the z-shift for anisotropic data.
    """
    ndim = layer.ndim
    n = ndim + 1

    # physical->world: the user-set affine (updated by GUI transform mode)
    M_affine = np.array(layer.affine.affine_matrix)  # (n x n)

    # data->physical: voxel scale and translate (set by layer.scale / .translate)
    M_d2p = np.eye(n)
    M_d2p[:ndim, :ndim] = np.diag(np.asarray(layer.scale))
    M_d2p[:ndim,  ndim] = np.asarray(layer.translate)

    return M_affine @ M_d2p


def decompose_layer_transform(layer) -> dict:
    """
    Given a napari Image (or any) layer, extract and decompose its full
    data->world transform using public API properties only.

    See _full_affine_matrix() for why we compose from layer.affine,
    layer.scale, and layer.translate rather than reading a single property.
    """
    ndim = layer.ndim
    matrix = _full_affine_matrix(layer)

    if ndim == 2:
        return decompose_affine_2d(matrix)
    elif ndim == 3:
        return decompose_affine_3d(matrix)
    else:
        raise ValueError(f"Only 2D and 3D layers supported, got ndim={ndim}")


def relative_transform(ref_layer, mov_layer) -> np.ndarray:
    """
    Compute the affine matrix that maps moving data coords -> reference data coords.
    T_rel = inv(T_ref_full) @ T_mov_full

    Uses _full_affine_matrix() for both layers so that layer.scale and
    layer.translate are included alongside the GUI-set layer.affine.
    A pure z-translation set via layer.translate is therefore correctly
    captured and will be reproduced by _apply_transform_and_crop().
    """
    T_ref = _full_affine_matrix(ref_layer)
    T_mov = _full_affine_matrix(mov_layer)
    return np.linalg.inv(T_ref) @ T_mov


# ---------------------------------------------------------------------------
# Napari widget (magicgui-based, works as a dock widget)
# ---------------------------------------------------------------------------

def make_widget():
    """
    Returns a napari dock widget for rigid registration parameter extraction.
    The widget has two buttons:
      - "Extract Parameters" : reads transforms from all Image layers and
        prints them to the console; stores results internally.
      - "Save CSV..."        : opens a native file-save dialog and writes
        the last extracted results to a .csv file.

    Import and register this in your plugin's napari.yaml or call directly.
    """
    from magicgui import magicgui
    from magicgui.widgets import Container, PushButton, ComboBox, CheckBox, Label
    import napari
    from napari.layers import Image
    from napari import current_viewer
    from qtpy.QtWidgets import QFileDialog

    # ---- Sub-widgets -------------------------------------------------------
    ref_combo = ComboBox(label="Reference layer")
    rel_check = CheckBox(label="Parameters relative to reference", value=True)
    status_label = Label(value="")

    extract_btn = PushButton(text="Extract Parameters")
    save_btn    = PushButton(text="Save CSV…")
    save_btn.enabled = False  # disabled until a successful extraction

    # Shared state: last extraction result + detected ndim
    _state = {"results": None, "ndim": None}

    # Keep the reference combo in sync with viewer layers
    def _refresh_layers(*_):
        """Rebuild the combo choices from current Image layers.

        Preserves the previously selected layer name if it still exists,
        so that adding/removing an unrelated layer doesn't silently change
        the user's reference choice.
        """
        viewer = current_viewer()
        if viewer is None:
            return
        names = [l.name for l in viewer.layers if isinstance(l, Image)]
        previous = ref_combo.value  # remember current selection
        ref_combo.choices = names
        if names:
            # Keep previous selection if still present, else fall back to first
            ref_combo.value = previous if previous in names else names[0]

    def _connect_layer_name_event(layer):
        """Connect _refresh_layers to a single layer's name-change event."""
        try:
            layer.events.name.connect(_refresh_layers)
        except AttributeError:
            pass

    def _disconnect_layer_name_event(layer):
        try:
            layer.events.name.disconnect(_refresh_layers)
        except Exception:
            pass

    def _on_layer_inserted(event):
        _connect_layer_name_event(event.value)
        _refresh_layers()

    def _on_layer_removed(event):
        _disconnect_layer_name_event(event.value)
        _refresh_layers()

    def _connect_viewer_events():
        """Hook into napari's layer-list signals once the viewer is available."""
        viewer = current_viewer()
        if viewer is None:
            return
        viewer.layers.events.inserted.connect(_on_layer_inserted)
        viewer.layers.events.removed.connect(_on_layer_removed)
        viewer.layers.events.reordered.connect(_refresh_layers)
        # Connect name events for layers already present
        for layer in viewer.layers:
            _connect_layer_name_event(layer)
        _refresh_layers()  # populate immediately

    # ---- Extract callback --------------------------------------------------
    def _on_extract():
        viewer = current_viewer()
        if viewer is None:
            status_label.value = "⚠ No active viewer."
            return

        image_layers = [l for l in viewer.layers if isinstance(l, Image)]
        if not image_layers:
            status_label.value = "⚠ No Image layers found."
            return

        ref_layer = next(
            (l for l in image_layers if l.name == ref_combo.value), None
        )
        if ref_layer is None:
            status_label.value = "⚠ Reference layer not found."
            return

        results = {}
        ndim    = image_layers[0].ndim

        for layer in image_layers:
            name = layer.name
            if rel_check.value and layer is not ref_layer:
                rel_matrix = relative_transform(ref_layer, layer)
                params = (decompose_affine_2d(rel_matrix) if ndim == 2
                          else decompose_affine_3d(rel_matrix))
                params["note"] = f"relative to '{ref_layer.name}'"
            else:
                params = decompose_layer_transform(layer)
                params["note"] = "absolute (world coords)"

            results[name] = params
            _print_params(name, params, layer.ndim)

        _state["results"] = results
        _state["ndim"]    = ndim
        save_btn.enabled  = True
        n = len(results)
        status_label.value = f"✓ Extracted {n} layer{'s' if n != 1 else ''}. Ready to save."

    # ---- Save CSV callback -------------------------------------------------
    def _on_save():
        if not _state["results"]:
            status_label.value = "⚠ Nothing to save — extract first."
            return

        path, _ = QFileDialog.getSaveFileName(
            None,
            "Save registration parameters as CSV",
            "registration_params.csv",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return  # user cancelled

        _export(_state["results"], path, "CSV", _state["ndim"])
        status_label.value = f"✓ Saved → {Path(path).name}"

    extract_btn.changed.connect(_on_extract)
    save_btn.changed.connect(_on_save)

    container = Container(
        widgets=[ref_combo, rel_check, extract_btn, save_btn, status_label],
        labels=False,
    )

    # Connect to viewer events once Qt is running (viewer may not exist yet
    # if the widget is created before a viewer is opened)
    import qtpy.QtCore as QtCore
    QtCore.QTimer.singleShot(200, _connect_viewer_events)

    return container


def _print_params(name: str, params: dict, ndim: int):
    """Pretty-print parameters to console."""
    print(f"\n{'='*60}")
    print(f"Layer: {name}")
    print(f"Note:  {params.get('note', '')}")
    if ndim == 2:
        ty, tx = params["translation_yx"]
        sy, sx = params["scale_yx"]
        print(f"  Translation (y, x):  ({ty:.3f}, {tx:.3f}) px")
        print(f"  Rotation:            {params['rotation_deg']:.4f} deg")
        print(f"  Scale    (y, x):     ({sy:.4f}, {sx:.4f})")
        if abs(params["shear"]) > 1e-3:
            print(f"  ⚠ Shear detected:   {params['shear']:.4f} (non-rigid!)")
    else:
        tz, ty, tx = params["translation_zyx"]
        sz, sy, sx = params["scale_zyx"]
        rx, ry, rz = params["rotation_euler_xyz_deg"]
        print(f"  Translation (z,y,x): ({tz:.3f}, {ty:.3f}, {tx:.3f}) vox")
        print(f"  Rotation XYZ (deg):  ({rx:.4f}, {ry:.4f}, {rz:.4f})")
        print(f"  Scale    (z,y,x):    ({sz:.4f}, {sy:.4f}, {sx:.4f})")
    print(f"{'='*60}")


def _export(results: dict, path: str, fmt: str, ndim: int):
    """Write results to a CSV file at *path* (extension forced to .csv)."""
    import csv
    out = Path(path).with_suffix(".csv")
    with open(out, "w", newline="") as f:
        if ndim == 2:
            fieldnames = ["layer", "note",
                          "translation_y_px", "translation_x_px",
                          "rotation_deg",
                          "scale_y", "scale_x", "shear"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, p_ in results.items():
                ty, tx = p_["translation_yx"]
                sy, sx = p_["scale_yx"]
                writer.writerow({
                    "layer": name,
                    "note": p_.get("note", ""),
                    "translation_y_px": ty,
                    "translation_x_px": tx,
                    "rotation_deg": p_["rotation_deg"],
                    "scale_y": sy,
                    "scale_x": sx,
                    "shear": p_.get("shear", 0.0),
                })
        else:
            fieldnames = ["layer", "note",
                          "translation_z_vox", "translation_y_vox", "translation_x_vox",
                          "rotation_euler_x_deg", "rotation_euler_y_deg", "rotation_euler_z_deg",
                          "scale_z", "scale_y", "scale_x"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, p_ in results.items():
                tz, ty, tx = p_["translation_zyx"]
                rx, ry, rz = p_["rotation_euler_xyz_deg"]
                sz, sy, sx = p_["scale_zyx"]
                writer.writerow({
                    "layer": name,
                    "note": p_.get("note", ""),
                    "translation_z_vox": tz,
                    "translation_y_vox": ty,
                    "translation_x_vox": tx,
                    "rotation_euler_x_deg": rx,
                    "rotation_euler_y_deg": ry,
                    "rotation_euler_z_deg": rz,
                    "scale_z": sz,
                    "scale_y": sy,
                    "scale_x": sx,
                })
    print(f"Saved CSV -> {out}")

def extract_all_transforms(viewer, reference_name: str = None) -> dict:
    """
    Programmatic API: extract transforms from all Image layers in a viewer.

    Parameters
    ----------
    viewer : napari.Viewer
    reference_name : str, optional
        Layer name to treat as reference. If None, returns absolute transforms.

    Returns
    -------
    dict mapping layer_name -> parameter dict
    """
    from napari.layers import Image

    image_layers = [l for l in viewer.layers if isinstance(l, Image)]
    ref = None
    if reference_name:
        matches = [l for l in image_layers if l.name == reference_name]
        if not matches:
            raise ValueError(f"Reference layer '{reference_name}' not found.")
        ref = matches[0]

    results = {}
    for layer in image_layers:
        if ref and layer is not ref:
            rel_matrix = relative_transform(ref, layer)
            ndim = layer.ndim
            params = decompose_affine_2d(rel_matrix) if ndim == 2 else decompose_affine_3d(rel_matrix)
            params["note"] = f"relative to '{ref.name}'"
        else:
            params = decompose_layer_transform(layer)
            params["note"] = "absolute"
        results[layer.name] = params

    return results


# ---------------------------------------------------------------------------
# Plugin registration entry point
# ---------------------------------------------------------------------------

def napari_experimental_provide_dock_widget():
    """
    Hook for napari plugin system (npe1 / contributions key for npe2).
    Register this in napari.yaml:

        contributions:
          widgets:
            - name: Rigid Registration Extractor
              command: rigid_registration_plugin.napari_experimental_provide_dock_widget
              display_name: Rigid Registration Extractor
    """
    return make_widget, {"name": "Rigid Registration Extractor"}


# ---------------------------------------------------------------------------
# Quick-start: run directly to open napari with the widget attached
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Widget 2: Apply transforms → crop → concatenate → 4D stack
# ---------------------------------------------------------------------------

def _build_affine_from_csv_row_3d(row: dict) -> np.ndarray:
    """
    Reconstruct the 4x4 affine matrix from a CSV row written by _export().

    The CSV stores decomposed parameters (translation, Euler angles, scale).
    We reassemble: M = T @ R @ S  (translation · rotation · scale),
    which is the standard TRS convention used when decomposing via SVD.

    Parameters are expected in ZYX voxel space (napari convention).
    """
    from scipy.spatial.transform import Rotation

    tz = float(row["translation_z_vox"])
    ty = float(row["translation_y_vox"])
    tx = float(row["translation_x_vox"])

    rx = float(row["rotation_euler_x_deg"])
    ry = float(row["rotation_euler_y_deg"])
    rz = float(row["rotation_euler_z_deg"])

    sz = float(row["scale_z"])
    sy = float(row["scale_y"])
    sx = float(row["scale_x"])

    R = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()  # 3×3
    S = np.diag([sz, sy, sx])                                                # 3×3

    M = np.eye(4)
    M[:3, :3] = R @ S
    M[0, 3] = tz
    M[1, 3] = ty
    M[2, 3] = tx
    return M


def _apply_transform_and_crop(
    data: np.ndarray,
    affine: np.ndarray,
    ref_shape: tuple,
    order: int = 1,
    z_pad: int = 0,
) -> np.ndarray:
    """
    Warp *data* with *affine* (4×4, ZYX, data→reference) and crop to output shape.

    scipy.ndimage.affine_transform pulls from INPUT space at each OUTPUT voxel,
    so it needs the *inverse* mapping: output_coord -> input_coord.

    The correct decomposition of the inverse is:

        p_in = A_inv @ p_ref
             = R_inv @ S_inv @ (p_ref - t)        (TRS convention)

    so:
        matrix = R_inv @ S_inv  =  inv(R @ S)
        offset = -matrix @ t

    We derive matrix and offset this way rather than inverting the full 4×4,
    which would silently fold the translation into the linear part when
    scale ≠ 1, corrupting the z (and y/x) shift.

    Z-padding shifts the output origin by z_pad slices:
        offset_padded = offset + matrix @ [z_pad, 0, 0]

    Parameters
    ----------
    data      : 3-D array (Z, Y, X)
    affine    : 4×4 homogeneous affine (data → reference space)
    ref_shape : (Z, Y, X) shape of the reference volume (before padding)
    order     : spline interpolation order (0=nearest, 1=linear, 3=cubic)
    z_pad     : number of slices to add above AND below the reference z-extent

    Returns
    -------
    warped : 3-D array with shape (ref_shape[0] + 2*z_pad, ref_shape[1], ref_shape[2])
    """
    from scipy.ndimage import affine_transform

    out_shape = (ref_shape[0] + 2 * z_pad, ref_shape[1], ref_shape[2])

    # Decompose the forward affine into linear (A) and translation (t) parts.
    A = affine[:3, :3]   # R @ S  (rotation composed with scale)
    t = affine[:3,  3]   # translation vector  [tz, ty, tx]

    # Inverse linear part: maps reference coords back to data coords
    A_inv = np.linalg.inv(A)

    # Correct offset:  maps reference origin (0,0,0) back to data space
    offset = -A_inv @ t

    # Account for z-padding: output voxel [0,y,x] corresponds to reference
    # voxel [-z_pad, y, x], so we shift the origin by z_pad in output space.
    if z_pad:
        offset = offset + A_inv @ np.array([z_pad, 0.0, 0.0])

    warped = affine_transform(
        data,
        matrix=A_inv,
        offset=offset,
        output_shape=out_shape,
        order=order,
        mode="constant",
        cval=0.0,
    )
    return warped.astype(data.dtype)


def make_apply_widget():
    """
    Napari dock widget — Widget 2: Apply & Stack.

    Workflow
    --------
    1. Load the CSV saved by Widget 1 (via "Browse CSV…").
    2. Select the reference layer from the dropdown (must be loaded in viewer).
    3. Choose interpolation order.
    4. Click "Apply & Build 4D Stack".

    The widget will:
      a. Read the affine for every layer named in the CSV.
      b. For layers whose note says "relative to <ref>", use the matrix as-is.
         For layers listed as "absolute", compute T_rel = inv(T_ref) @ T_mov.
      c. Warp each volume into reference space and crop to the reference shape.
      d. Stack all volumes (reference first, then timepoints in CSV order) into
         a 4D array (T, Z, Y, X) and add it to the viewer as a new Image layer.
    """
    import csv
    from magicgui.widgets import Container, PushButton, ComboBox, CheckBox, Label, SpinBox
    from napari.layers import Image
    from napari import current_viewer
    from qtpy.QtWidgets import QFileDialog

    # ── Sub-widgets ──────────────────────────────────────────────────────────
    csv_label   = Label(value="No CSV loaded")
    browse_btn  = PushButton(text="Browse CSV…")

    ref_combo   = ComboBox(label="Reference layer")
    interp_spin = SpinBox(label="Interpolation order (0–3)", value=1, min=0, max=3)
    z_pad_spin  = SpinBox(label="Z padding (slices, each side)", value=0, min=0, max=500)

    apply_btn   = PushButton(text="Apply & Build 4D Stack")
    status_label = Label(value="")

    _state = {"csv_rows": None, "csv_path": None}

    # ── Layer list refresh ───────────────────────────────────────────────────
    def _refresh_layers(*_):
        viewer = current_viewer()
        if viewer is None:
            return
        names = [l.name for l in viewer.layers if isinstance(l, Image)]
        prev = ref_combo.value
        ref_combo.choices = names
        if names:
            ref_combo.value = prev if prev in names else names[0]

    def _connect_layer_name_event(layer):
        try:
            layer.events.name.connect(_refresh_layers)
        except AttributeError:
            pass

    def _disconnect_layer_name_event(layer):
        try:
            layer.events.name.disconnect(_refresh_layers)
        except Exception:
            pass

    def _on_layer_inserted(event):
        _connect_layer_name_event(event.value)
        _refresh_layers()

    def _on_layer_removed(event):
        _disconnect_layer_name_event(event.value)
        _refresh_layers()

    def _connect_viewer_events():
        viewer = current_viewer()
        if viewer is None:
            return
        viewer.layers.events.inserted.connect(_on_layer_inserted)
        viewer.layers.events.removed.connect(_on_layer_removed)
        viewer.layers.events.reordered.connect(_refresh_layers)
        for layer in viewer.layers:
            _connect_layer_name_event(layer)
        _refresh_layers()

    # ── CSV browse ───────────────────────────────────────────────────────────
    def _on_browse():
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Open registration parameters CSV",
            "",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return

        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            status_label.value = "⚠ CSV is empty."
            return

        # Validate expected columns
        required = {
            "layer",
            "translation_z_vox", "translation_y_vox", "translation_x_vox",
            "rotation_euler_x_deg", "rotation_euler_y_deg", "rotation_euler_z_deg",
            "scale_z", "scale_y", "scale_x",
        }
        missing = required - set(rows[0].keys())
        if missing:
            status_label.value = f"⚠ CSV missing columns: {missing}"
            return

        _state["csv_rows"] = rows
        _state["csv_path"] = path
        csv_label.value = f"CSV: {Path(path).name}  ({len(rows)} rows)"
        status_label.value = ""

    # ── Apply & stack ────────────────────────────────────────────────────────
    def _on_apply():
        viewer = current_viewer()
        if viewer is None:
            status_label.value = "⚠ No active viewer."
            return
        if not _state["csv_rows"]:
            status_label.value = "⚠ Load a CSV first."
            return

        image_layers = {l.name: l for l in viewer.layers if isinstance(l, Image)}
        ref_name = ref_combo.value
        if ref_name not in image_layers:
            status_label.value = f"⚠ Reference layer '{ref_name}' not in viewer."
            return

        ref_layer = image_layers[ref_name]
        ref_data  = np.asarray(ref_layer.data)
        if ref_data.ndim != 3:
            status_label.value = "⚠ Reference must be a 3D volume."
            return
        ref_shape = ref_data.shape  # (Z, Y, X) — unpadded

        # Reference identity affine (it maps to itself)
        ref_affine = np.eye(4)

        order  = interp_spin.value
        z_pad  = z_pad_spin.value
        frames = []   # will hold (timepoint_label, warped_array)
        errors = []

        # ── Determine CSV row ordering ────────────────────────────────────
        # The reference row (if present) is treated as t=0; remaining rows
        # follow the CSV order.  If the reference is absent from the CSV
        # (it was saved as "absolute / world"), it is still prepended.
        rows      = _state["csv_rows"]
        ref_row   = next((r for r in rows if r["layer"] == ref_name), None)
        other_rows = [r for r in rows if r["layer"] != ref_name]

        # Prepend reference — pad in z if requested
        if z_pad > 0:
            pad_slices = np.zeros(
                (z_pad, ref_shape[1], ref_shape[2]), dtype=ref_data.dtype
            )
            ref_padded = np.concatenate([pad_slices, ref_data, pad_slices], axis=0)
        else:
            ref_padded = ref_data
        frames.append((ref_name, ref_padded))

        for row in other_rows:
            layer_name = row["layer"]
            note       = row.get("note", "")

            if layer_name not in image_layers:
                errors.append(f"'{layer_name}' not found in viewer — skipped")
                continue

            mov_data = np.asarray(image_layers[layer_name].data)
            if mov_data.ndim != 3:
                errors.append(f"'{layer_name}' is not 3D — skipped")
                continue

            # ── Reconstruct affine ────────────────────────────────────────
            # If the CSV note says "relative to '<ref>'", the stored matrix
            # already expresses mov→ref.  If it says "absolute", we need to
            # re-derive relative: T_rel = inv(T_ref_world) @ T_mov_world.
            # Since Widget 1 always saves relative transforms when the
            # checkbox is ticked (default), the common case is relative.
            if f"relative to '{ref_name}'" in note:
                affine = _build_affine_from_csv_row_3d(row)
            else:
                # Absolute: reconstruct world affines and relativise
                # Use full data->world (includes scale) for both layers
                T_ref_world = np.array(ref_layer._transforms.simplified.affine_matrix)
                T_mov_world = np.array(image_layers[layer_name]._transforms.simplified.affine_matrix)
                affine = np.linalg.inv(T_ref_world) @ T_mov_world

            status_label.value = f"Warping '{layer_name}'…"

            try:
                warped = _apply_transform_and_crop(
                    mov_data, affine, ref_shape, order=order, z_pad=z_pad
                )
                frames.append((layer_name, warped))
            except Exception as exc:
                errors.append(f"'{layer_name}' warp failed: {exc}")

        if len(frames) < 2:
            status_label.value = "⚠ Need at least 2 volumes to build a stack."
            if errors:
                print("\n".join(errors))
            return

        # ── Concatenate → 4D ─────────────────────────────────────────────
        stack = np.stack([f[1] for f in frames], axis=0)  # (T, Z, Y, X)
        labels = [f[0] for f in frames]

        viewer.add_image(
            stack,
            name="4D_registered_stack",
            colormap="gray",
            blending="additive",
        )

        summary = (
            f"✓ Stack: {stack.shape[0]}t × {stack.shape[1]}z "
            f"× {stack.shape[2]}y × {stack.shape[3]}x"
        )
        status_label.value = summary
        print("\n" + "="*60)
        print("4D stack built:")
        for i, lbl in enumerate(labels):
            print(f"  t={i:02d}  {lbl}")
        if errors:
            print("\nWarnings:")
            for e in errors:
                print(f"  ⚠ {e}")
        print("="*60)

    # ── Wire up ──────────────────────────────────────────────────────────────
    browse_btn.changed.connect(_on_browse)
    apply_btn.changed.connect(_on_apply)

    container = Container(
        widgets=[
            browse_btn, csv_label,
            ref_combo,
            interp_spin,
            z_pad_spin,
            apply_btn,
            status_label,
        ],
        labels=False,
    )

    import qtpy.QtCore as QtCore
    QtCore.QTimer.singleShot(200, _connect_viewer_events)

    return container


def napari_experimental_provide_dock_widget_apply():
    """Plugin hook for Widget 2 (Apply & Stack)."""
    return make_apply_widget, {"name": "Apply & Build 4D Stack"}


# ---------------------------------------------------------------------------
# Widget 3: Scale editor for the selected layer
# ---------------------------------------------------------------------------

def make_scale_widget():
    """
    Napari dock widget — Widget 3: Layer Scale Editor.

    Shows one FloatSpinBox per dimension for the currently selected layer,
    labelled with the viewer's axis labels (e.g. z, y, x).  Editing a value
    and pressing Enter (or clicking away) applies it immediately to the layer
    via layer.scale, which updates the world-space extent and rerenders.

    The widget refreshes its fields whenever the layer selection changes or
    a layer is added/removed, and reloads current values whenever the active
    layer's scale is changed externally (e.g. from the console).
    """
    from magicgui.widgets import Container, FloatSpinBox, Label, ComboBox
    from napari import current_viewer
    from napari.layers import Layer

    status_label  = Label(value="Select a layer to edit its scale.")
    fields_container = Container(widgets=[], labels=True)

    # Internal state: which layer we're currently watching
    _state = {"layer": None, "spinboxes": [], "blocking": False}

    def _make_spinboxes(ndim: int, axis_labels: tuple, current_scale: tuple):
        """Rebuild the per-axis FloatSpinBox widgets for an ndim layer."""
        spinboxes = []
        for i in range(ndim):
            label = axis_labels[i] if i < len(axis_labels) else f"axis-{i}"
            sb = FloatSpinBox(
                label=label,
                value=float(current_scale[i]),
                min=1e-6,
                max=1e6,
                step=0.1,
            )
            spinboxes.append(sb)
        return spinboxes

    def _rebuild_fields(layer):
        """Tear down old spinboxes and build fresh ones for *layer*."""
        viewer = current_viewer()

        # Disconnect old scale-change listener if any
        old = _state["layer"]
        if old is not None:
            try:
                old.events.scale.disconnect(_on_layer_scale_changed)
            except Exception:
                pass

        _state["layer"] = layer
        _state["spinboxes"] = []

        # Clear the fields container
        while len(fields_container) > 0:
            fields_container.pop(-1)

        if layer is None:
            status_label.value = "Select a layer to edit its scale."
            return

        ndim = layer.ndim
        # Axis labels come from the viewer dims (may be shorter than ndim for
        # layers with more dims than currently displayed — pad with indices)
        raw_labels = list(viewer.dims.axis_labels) if viewer else []
        # Align to the last ndim labels (napari broadcasts from trailing dims)
        if len(raw_labels) >= ndim:
            axis_labels = raw_labels[-ndim:]
        else:
            axis_labels = [f"axis-{i}" for i in range(ndim - len(raw_labels))] + raw_labels

        spinboxes = _make_spinboxes(ndim, axis_labels, layer.scale)
        _state["spinboxes"] = spinboxes

        for sb in spinboxes:
            fields_container.append(sb)
            sb.changed.connect(_on_spinbox_changed)

        # Watch for external scale changes (e.g. from console)
        layer.events.scale.connect(_on_layer_scale_changed)
        status_label.value = f"Editing: {layer.name}"

    def _on_spinbox_changed(value):
        """User edited a spinbox — push the new scale tuple to the layer."""
        if _state["blocking"]:
            return
        layer = _state["layer"]
        if layer is None:
            return
        new_scale = tuple(float(sb.value) for sb in _state["spinboxes"])
        _state["blocking"] = True
        try:
            layer.scale = new_scale
        finally:
            _state["blocking"] = False

    def _on_layer_scale_changed(event=None):
        """Layer scale changed externally — sync spinboxes without re-triggering."""
        if _state["blocking"]:
            return
        layer = _state["layer"]
        if layer is None:
            return
        _state["blocking"] = True
        try:
            for sb, val in zip(_state["spinboxes"], layer.scale):
                sb.value = float(val)
        finally:
            _state["blocking"] = False

    def _on_selection_changed(event=None):
        """Viewer layer selection changed — switch to the newly active layer."""
        viewer = current_viewer()
        if viewer is None:
            return
        active = viewer.layers.selection.active
        _rebuild_fields(active)

    def _connect_viewer_events():
        viewer = current_viewer()
        if viewer is None:
            return
        viewer.layers.selection.events.active.connect(_on_selection_changed)
        viewer.layers.events.inserted.connect(_on_selection_changed)
        viewer.layers.events.removed.connect(_on_selection_changed)
        # Populate immediately with whatever is already selected
        _on_selection_changed()

    container = Container(
        widgets=[status_label, fields_container],
        labels=False,
    )

    import qtpy.QtCore as QtCore
    QtCore.QTimer.singleShot(200, _connect_viewer_events)

    return container


def napari_experimental_provide_dock_widget_scale():
    """Plugin hook for Widget 3 (Scale Editor)."""
    return make_scale_widget, {"name": "Layer Scale Editor"}



# ---------------------------------------------------------------------------
# Widget 4: Merge two registration CSV files by image name
# ---------------------------------------------------------------------------

def make_merge_csv_widget():
    """
    Napari dock widget — Widget 4: Merge Registration CSVs.

    Use case
    --------
    You have aligned the same set of images in two separate sessions (e.g.
    first aligning XY, then fine-tuning Z), each producing a CSV.  This
    widget merges them into a single CSV, combining the transform parameters
    per image by composing the two affine matrices:

        M_combined = M_b @ M_a   (apply A first, then B on top)

    The image name ("layer" column) is used as the join key.  Rows present
    in only one file are passed through unchanged.  The merged result is
    decomposed back into translation / rotation / scale columns so it is
    directly usable by Widget 2.

    Merge modes
    -----------
    compose : M_combined = M_b @ M_a  — stack both transforms (default)
    prefer_a: keep CSV A row, ignore CSV B for that image
    prefer_b: keep CSV B row, ignore CSV A for that image
    """
    import csv
    from magicgui.widgets import Container, PushButton, ComboBox, Label
    from qtpy.QtWidgets import QFileDialog

    # ── Sub-widgets ──────────────────────────────────────────────────────────
    label_a    = Label(value="CSV A: not loaded")
    browse_a   = PushButton(text="Browse CSV A…")
    label_b    = Label(value="CSV B: not loaded")
    browse_b   = PushButton(text="Browse CSV B…")
    mode_combo = ComboBox(
        label="Merge mode",
        choices=["compose", "prefer_a", "prefer_b"],
        value="compose",
    )
    merge_btn  = PushButton(text="Merge & Save CSV…")
    status     = Label(value="")

    _state = {"rows_a": None, "rows_b": None}

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _load_csv(path: str):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError("CSV is empty")
        required = {
            "layer",
            "translation_z_vox", "translation_y_vox", "translation_x_vox",
            "rotation_euler_x_deg", "rotation_euler_y_deg", "rotation_euler_z_deg",
            "scale_z", "scale_y", "scale_x",
        }
        missing = required - set(rows[0].keys())
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return rows

    def _row_to_matrix(row: dict) -> np.ndarray:
        """Reconstruct the 4×4 affine from a CSV row (same as Widget 2)."""
        return _build_affine_from_csv_row_3d(row)

    def _matrix_to_row(name: str, M: np.ndarray, note: str = "") -> dict:
        """Decompose a 4×4 affine back to CSV columns."""
        params = decompose_affine_3d(M)
        tz, ty, tx       = params["translation_zyx"]
        rx, ry, rz       = params["rotation_euler_xyz_deg"]
        sz, sy, sx       = params["scale_zyx"]
        return {
            "layer":                name,
            "note":                 note,
            "translation_z_vox":   tz,
            "translation_y_vox":   ty,
            "translation_x_vox":   tx,
            "rotation_euler_x_deg": rx,
            "rotation_euler_y_deg": ry,
            "rotation_euler_z_deg": rz,
            "scale_z":             sz,
            "scale_y":             sy,
            "scale_x":             sx,
        }

    def _write_csv(rows: list, path: str):
        import csv as _csv
        fieldnames = [
            "layer", "note",
            "translation_z_vox", "translation_y_vox", "translation_x_vox",
            "rotation_euler_x_deg", "rotation_euler_y_deg", "rotation_euler_z_deg",
            "scale_z", "scale_y", "scale_x",
        ]
        with open(path, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    # ── Browse callbacks ─────────────────────────────────────────────────────
    def _on_browse_a():
        path, _ = QFileDialog.getOpenFileName(
            None, "Open CSV A", "", "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            _state["rows_a"] = _load_csv(path)
            label_a.value = f"CSV A: {Path(path).name}  ({len(_state['rows_a'])} rows)"
            status.value = ""
        except Exception as e:
            status.value = f"⚠ CSV A: {e}"

    def _on_browse_b():
        path, _ = QFileDialog.getOpenFileName(
            None, "Open CSV B", "", "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            _state["rows_b"] = _load_csv(path)
            label_b.value = f"CSV B: {Path(path).name}  ({len(_state['rows_b'])} rows)"
            status.value = ""
        except Exception as e:
            status.value = f"⚠ CSV B: {e}"

    # ── Merge callback ───────────────────────────────────────────────────────
    def _on_merge():
        if not _state["rows_a"] or not _state["rows_b"]:
            status.value = "⚠ Load both CSV files first."
            return

        path, _ = QFileDialog.getSaveFileName(
            None, "Save merged CSV", "merged_registration.csv",
            "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return

        mode = mode_combo.value

        # Index both CSVs by layer name
        dict_a = {r["layer"]: r for r in _state["rows_a"]}
        dict_b = {r["layer"]: r for r in _state["rows_b"]}
        all_names = list(dict_a.keys())  # preserve A's ordering
        for name in dict_b:             # append B-only entries at the end
            if name not in dict_a:
                all_names.append(name)

        merged_rows = []
        report = []

        for name in all_names:
            in_a = name in dict_a
            in_b = name in dict_b

            if in_a and in_b:
                if mode == "compose":
                    # Stack: apply A first, B on top → M = M_b @ M_a
                    M_a = _row_to_matrix(dict_a[name])
                    M_b = _row_to_matrix(dict_b[name])
                    M_c = M_b @ M_a
                    note = "composed: B @ A"
                    row  = _matrix_to_row(name, M_c, note)
                elif mode == "prefer_a":
                    row  = dict(dict_a[name])
                    row["note"] = "prefer_a"
                else:  # prefer_b
                    row  = dict(dict_b[name])
                    row["note"] = "prefer_b"
                report.append(f"  {name}: {'composed' if mode == 'compose' else mode}")

            elif in_a:
                row = dict(dict_a[name])
                report.append(f"  {name}: A only (pass-through)")
            else:
                row = dict(dict_b[name])
                report.append(f"  {name}: B only (pass-through)")

            merged_rows.append(row)

        _write_csv(merged_rows, path)

        n = len(merged_rows)
        status.value = f"✓ Saved {n} rows → {Path(path).name}"
        print("\n" + "="*60)
        print(f"Merged CSV ({mode} mode) → {path}")
        print("\n".join(report))
        print("="*60)

    # ── Wire up ──────────────────────────────────────────────────────────────
    browse_a.changed.connect(_on_browse_a)
    browse_b.changed.connect(_on_browse_b)
    merge_btn.changed.connect(_on_merge)

    return Container(
        widgets=[browse_a, label_a, browse_b, label_b, mode_combo, merge_btn, status],
        labels=False,
    )


def napari_experimental_provide_dock_widget_merge():
    """Plugin hook for Widget 4 (Merge CSVs)."""
    return make_merge_csv_widget, {"name": "Merge Registration CSVs"}

if __name__ == "__main__":
    import napari
    import numpy as np

    viewer = napari.Viewer()

    # # --- Demo: add two synthetic 3D volumes ---
    # rng = np.random.default_rng(42)
    # ref_vol  = rng.random((30, 128, 128)).astype(np.float32)
    # mov_vol  = rng.random((30, 128, 128)).astype(np.float32)
    # mov_vol2 = rng.random((30, 128, 128)).astype(np.float32)
    #
    # viewer.add_image(ref_vol,  name="t0_reference", colormap="green",  opacity=0.6)
    # viewer.add_image(mov_vol,  name="t1_moving",    colormap="magenta", opacity=0.6)
    # viewer.add_image(mov_vol2, name="t2_moving",    colormap="cyan",    opacity=0.6)

    print("="*60)
    print("How to use:")
    print("  1. Double-click a layer name to activate transform handles")
    print("  2. Drag / rotate the layer to align it")
    print("  3. Use the 'Rigid Registration Extractor' dock widget")
    print("     to extract and export the parameters")
    print("="*60)

    w1, m1 = napari_experimental_provide_dock_widget()
    viewer.window.add_dock_widget(w1(), name=m1["name"])

    w2, m2 = napari_experimental_provide_dock_widget_apply()
    viewer.window.add_dock_widget(w2(), name=m2["name"])

    w3, m3 = napari_experimental_provide_dock_widget_scale()
    viewer.window.add_dock_widget(w3(), name=m3["name"])

    w4, m4 = napari_experimental_provide_dock_widget_merge()
    viewer.window.add_dock_widget(w4(), name=m4["name"])

    # Open the built-in IPython console so the user can inspect layers,
    # transforms, and the output stack directly.  napari pre-injects `viewer`
    # into the console namespace, so viewer.layers, np, etc. work immediately.
    # _show_key_bindings_dialog -> show() is the public way to reveal it.
    viewer.window.qt_viewer.console.show()

    napari.run()
