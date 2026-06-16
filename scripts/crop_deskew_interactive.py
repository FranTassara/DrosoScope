"""
Interactive post-processing: Crop + Deskew
==========================================

Interactive GUI tool to crop and deskew raw OPM data.

Workflow:
1. Select the folder containing data_raw_*.tif files.
2. Choose the raw data orientation (4 options shown for comparison).
3. Inspect the max projection with the selected orientation.
4. Draw a rectangle to define the region of interest (ROI).
5. Confirm optical parameters (angle, galvo step, pixel size).
6. Crop + deskew is applied to all files in the folder.

Usage:
    python scripts/crop_deskew_interactive.py
"""

import os
import sys
import glob
import numpy as np
import tifffile
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, RadioButtons

# ──────────────────────────────────────────────────────────────────────────────
# Imports del proyecto
# ──────────────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # DrosoScope root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from postprocessing.deskew import deskew, max_projection_z

try:
    from config import DEFAULT_CONFIG
    _TILT_DEG   = DEFAULT_CONFIG['tilt_deg']
    _GALVO_STEP = DEFAULT_CONFIG['galvo_step_um']
    _PIXEL_SIZE = DEFAULT_CONFIG['sample_px_um']
except ImportError:
    _TILT_DEG   = 41.0
    _GALVO_STEP = 0.168
    _PIXEL_SIZE = 0.127


# ──────────────────────────────────────────────────────────────────────────────
# Orientación del raw
# ──────────────────────────────────────────────────────────────────────────────
# Cada opción es (etiqueta_corta, descripción, función: data(Z,*,*) → data(Z,H,W))
ORIENT_OPTIONS = [
    ('[0] Sin transformar',
     'El archivo ya está en (Z, H, W)',
     lambda d: d),
    ('[1] Transponer H↔W  ← pipeline normal',
     'El raw está en (Z, W, H) → se transpone a (Z, H, W)',
     lambda d: d.transpose(0, 2, 1)),
    ('[2] Espejo vertical (flip filas)',
     'Como [0] pero con filas invertidas (flip H)',
     lambda d: np.flip(d, axis=1)),
    ('[3] Espejo horizontal (flip columnas)',
     'Como [0] pero con columnas invertidas (flip W)',
     lambda d: np.flip(d, axis=2)),
]


def apply_orientation(data_3d, orient_idx):
    """Aplica la transformación de orientación elegida a un stack (Z, *, *)."""
    _, _, fn = ORIENT_OPTIONS[orient_idx]
    return fn(data_3d)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de UI
# ──────────────────────────────────────────────────────────────────────────────

def _tk_root():
    """Crea una ventana Tk oculta y la lleva al frente."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    return root


def ask_folder():
    root = _tk_root()
    folder = filedialog.askdirectory(
        title="Seleccioná la carpeta con las imágenes raw",
        parent=root,
    )
    root.destroy()
    return folder or None


def ask_params():
    """Dialogs para confirmar/editar los parámetros ópticos."""
    root = _tk_root()
    tilt = simpledialog.askfloat(
        "Parámetros ópticos",
        "Ángulo de inclinación (grados):",
        initialvalue=_TILT_DEG, minvalue=1.0, maxvalue=89.0,
        parent=root,
    )
    step = simpledialog.askfloat(
        "Parámetros ópticos",
        "Paso del galvo (µm):",
        initialvalue=_GALVO_STEP, minvalue=0.001,
        parent=root,
    )
    pixel = simpledialog.askfloat(
        "Parámetros ópticos",
        "Tamaño de píxel de la cámara (µm):",
        initialvalue=_PIXEL_SIZE, minvalue=0.001,
        parent=root,
    )
    root.destroy()
    return (
        tilt  if tilt  is not None else _TILT_DEG,
        step  if step  is not None else _GALVO_STEP,
        pixel if pixel is not None else _PIXEL_SIZE,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Selección de orientación
# ──────────────────────────────────────────────────────────────────────────────

def select_orientation(filepath):
    """Muestra las 4 orientaciones posibles del raw y devuelve el índice elegido.

    Se muestran las proyecciones máximas en un panel 2×2.
    El usuario elige con los radio buttons y cierra la ventana para confirmar.
    Por defecto está seleccionada la opción [1] (pipeline normal: Z,W,H → Z,H,W).
    """
    raw = tifffile.imread(filepath)
    if raw.ndim == 2:
        return 0   # imagen 2-D: no hay orientación que elegir

    # Calcular proyección máxima para cada orientación
    projs = []
    for _, _, fn in ORIENT_OPTIONS:
        p = fn(raw).max(axis=0).astype(np.float32)
        projs.append(p)

    chosen = [1]   # default: opción 1 (pipeline normal)

    fig = plt.figure(figsize=(15, 9))
    try:
        fig.canvas.manager.set_window_title('Orientación del raw — elegí y cerrá para confirmar')
    except Exception:
        pass

    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.32],
                          hspace=0.50, wspace=0.30)
    ax_imgs = [fig.add_subplot(gs[r, c])
               for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    ax_rb   = fig.add_subplot(gs[:, 2])

    def _highlight(active_idx):
        for j, ax in enumerate(ax_imgs):
            color = '#00d4ff' if j == active_idx else 'gray'
            lw    = 3         if j == active_idx else 0.8
            for sp in ax.spines.values():
                sp.set_edgecolor(color)
                sp.set_linewidth(lw)
        fig.canvas.draw_idle()

    for ax, proj, orient in zip(ax_imgs, projs, ORIENT_OPTIONS):
        short_lbl = orient[0]
        vmin = float(np.percentile(proj, 0.5))
        vmax = float(np.percentile(proj, 99.5))
        ax.imshow(proj, cmap='gray', vmin=vmin, vmax=vmax,
                  origin='upper', aspect='equal')
        ax.set_title(f'{short_lbl}\n{proj.shape[0]}×{proj.shape[1]} px (H×W)',
                     fontsize=9)
        ax.set_xlabel('columnas (W)')
        ax.set_ylabel('filas (H)')

    radio = RadioButtons(
        ax_rb,
        [f'{short_lbl}' for short_lbl, _, _ in ORIENT_OPTIONS],
        active=1,
    )
    # Make radio button labels a bit larger
    for lbl in radio.labels:
        lbl.set_fontsize(9)

    def _on_radio(label):
        idx = next(i for i, (s, _, _) in enumerate(ORIENT_OPTIONS) if s == label)
        chosen[0] = idx
        _highlight(idx)

    radio.on_clicked(_on_radio)
    _highlight(1)   # resaltar opción por defecto

    fig.suptitle(
        'Elegí la orientación correcta con los botones de la derecha\n'
        'Cerrá la ventana para confirmar',
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()

    return chosen[0]


# ──────────────────────────────────────────────────────────────────────────────
# Carga y proyección
# ──────────────────────────────────────────────────────────────────────────────

def load_for_preview(filepath, orient_idx=1):
    """Carga un raw .tif y devuelve la proyección máxima como (H, W).

    Aplica la orientación elegida antes de proyectar.
    """
    data = tifffile.imread(filepath)
    if data.ndim == 2:
        return data.astype(np.float32)
    data = apply_orientation(data, orient_idx)
    return data.max(axis=0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Selección de ROI
# ──────────────────────────────────────────────────────────────────────────────

def select_roi(max_proj):
    """Muestra la proyección máxima y devuelve el ROI seleccionado.

    Returns
    -------
    (x1, y1, x2, y2) : coordenadas en píxeles sobre la imagen (H, W)
        x → eje de columnas (ancho)
        y → eje de filas    (alto)
    None si no se seleccionó nada.
    """
    roi = [None]

    fig, ax = plt.subplots(figsize=(12, 9))
    try:
        fig.canvas.manager.set_window_title('Seleccionar ROI — cerrá para confirmar')
    except Exception:
        pass

    vmin = float(np.percentile(max_proj, 0.5))
    vmax = float(np.percentile(max_proj, 99.5))
    ax.imshow(max_proj, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal',
              origin='upper')
    ax.set_title(
        'Proyección máxima  ·  Dibujá el ROI con click + arrastrá\n'
        'Cerrá la ventana para confirmar',
        fontsize=11,
    )
    ax.set_xlabel('X — ancho (columnas)')
    ax.set_ylabel('Y — alto (filas)')

    def on_select(eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        h, w = max_proj.shape
        x1 = int(round(min(eclick.xdata, erelease.xdata)))
        y1 = int(round(min(eclick.ydata, erelease.ydata)))
        x2 = int(round(max(eclick.xdata, erelease.xdata)))
        y2 = int(round(max(eclick.ydata, erelease.ydata)))
        # clip a los límites de la imagen
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        roi[0] = (x1, y1, x2, y2)
        ax.set_title(
            f'ROI: x=[{x1}:{x2}]  y=[{y1}:{y2}]  '
            f'→  {y2 - y1} px alto × {x2 - x1} px ancho\n'
            'Cerrá la ventana para confirmar',
            fontsize=11,
        )
        fig.canvas.draw_idle()

    _selector = RectangleSelector(
        ax, on_select,
        useblit=True,
        button=[1],
        minspanx=10, minspany=10,
        spancoords='pixels',
        interactive=True,
    )

    plt.tight_layout()
    plt.show()      # bloquea hasta que el usuario cierre la ventana

    return roi[0]


# ──────────────────────────────────────────────────────────────────────────────
# Procesamiento de un archivo
# ──────────────────────────────────────────────────────────────────────────────

def process_file(filepath, roi, tilt_deg, galvo_step_um, pixel_size_um,
                 orient_idx=1):
    """Aplica crop y deskew a un archivo data_raw_*.tif.

    Pipeline:
        1. raw  →  apply_orientation(orient_idx)  →  images (Z, H, W)
        2. crop:  images[:, vstart:vend, hstart:hend]
        3. deskew prep:  flip(transpose(crop, (0,2,1)), axis=0)
        4. deskew  →  guardar deskew + maxZ
    """
    x1, y1, x2, y2 = roi
    hstart, hend = x1, x2   # columnas → ancho
    vstart, vend = y1, y2   # filas    → alto

    print(f"  Cargando: {os.path.basename(filepath)}")
    data = tifffile.imread(filepath)
    if data.ndim == 2:
        data = data[np.newaxis, ...]

    # Aplicar orientación elegida  →  (Z, H, W)
    images = apply_orientation(data, orient_idx)

    # ── Crop ──────────────────────────────────────────────────────────────────
    images_crop = images[:, vstart:vend, hstart:hend]
    out_dir  = os.path.dirname(filepath)
    basename = os.path.splitext(os.path.basename(filepath))[0]

    crop_name = basename.replace('data_raw_', 'data_crop_')
    crop_path = os.path.join(out_dir, crop_name + '.tif')
    tifffile.imwrite(crop_path, images_crop, imagej=True, metadata={'axes': 'ZYX'})
    print(f"    ✓ Crop   → {crop_name}.tif   {images_crop.shape}")

    # ── Deskew ────────────────────────────────────────────────────────────────
    images_for_deskew = np.flip(np.transpose(images_crop, (0, 2, 1)), axis=0)
    deskewed = deskew(
        data         = images_for_deskew,
        theta        = tilt_deg,
        distance     = galvo_step_um,
        pixel_size   = pixel_size_um,
        z_downsample = 2,
        output_dtype = 'uint16',
    )
    deskew_name = basename.replace('data_raw_', 'data_deskew_')
    deskew_path = os.path.join(out_dir, deskew_name + '.tif')
    tifffile.imwrite(deskew_path, deskewed, imagej=True, metadata={'axes': 'ZYX'})
    print(f"    ✓ Deskew → {deskew_name}.tif   {deskewed.shape}")

    # ── Max projection ────────────────────────────────────────────────────────
    maxz      = max_projection_z(deskewed)
    maxz_name = basename.replace('data_raw_', 'data_maxz_')
    maxz_path = os.path.join(out_dir, maxz_name + '.tif')
    tifffile.imwrite(maxz_path, maxz, imagej=True, metadata={'axes': 'YX'})
    print(f"    ✓ MaxZ   → {maxz_name}.tif   {maxz.shape}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Carpeta ----------------------------------------------------------------
    folder = ask_folder()
    if not folder:
        print("Cancelado: no se seleccionó ninguna carpeta.")
        return

    print(f"\nCarpeta: {folder}")

    # 2. Buscar archivos raw ----------------------------------------------------
    raw_files = sorted(glob.glob(os.path.join(folder, 'data_raw_*.tif')))
    if not raw_files:
        root = _tk_root()
        messagebox.showerror(
            "No se encontraron archivos",
            f"No hay archivos data_raw_*.tif en:\n{folder}",
            parent=root,
        )
        root.destroy()
        return

    print(f"\nArchivos raw encontrados ({len(raw_files)}):")
    for f in raw_files:
        print(f"  • {os.path.basename(f)}")

    # 3. Selección de orientación del raw --------------------------------------
    ref_file = raw_files[0]
    print(f"\nArchivo de referencia: {os.path.basename(ref_file)}")
    print("Se abrirá una ventana — elegí la orientación correcta y cerrá para continuar.")
    orient_idx = select_orientation(ref_file)
    print(f"  Orientación elegida: {ORIENT_OPTIONS[orient_idx][0]}")

    # 4. Proyección máxima con la orientación elegida --------------------------
    print(f"\nCalculando proyección máxima...")
    max_proj = load_for_preview(ref_file, orient_idx)
    print(f"  Tamaño: {max_proj.shape[1]} × {max_proj.shape[0]} px  (ancho × alto)")

    # 5. Selección de ROI -------------------------------------------------------
    print("\nSe abrirá una ventana — dibujá el ROI y cerrá la ventana para continuar.")
    roi = select_roi(max_proj)

    if roi is None:
        print("No se seleccionó ningún ROI. Saliendo.")
        return

    x1, y1, x2, y2 = roi
    print(f"\nROI confirmado: x=[{x1}:{x2}]  y=[{y1}:{y2}]")
    print(f"  Crop resultante: {y2 - y1} px alto × {x2 - x1} px ancho")

    # 6. Parámetros ópticos ----------------------------------------------------
    tilt_deg, galvo_step_um, pixel_size_um = ask_params()
    print(f"\nParámetros:")
    print(f"  Ángulo        : {tilt_deg} °")
    print(f"  Paso galvo    : {galvo_step_um} µm")
    print(f"  Tamaño píxel  : {pixel_size_um} µm")

    # 7. Procesar todos los archivos -------------------------------------------
    print(f"\nProcesando {len(raw_files)} archivo(s)...\n" + "-" * 50)
    errors = []
    for filepath in raw_files:
        try:
            process_file(filepath, roi, tilt_deg, galvo_step_um, pixel_size_um,
                         orient_idx)
        except Exception as exc:
            print(f"  ✗ ERROR en {os.path.basename(filepath)}: {exc}")
            errors.append((os.path.basename(filepath), exc))

    # 8. Resumen ---------------------------------------------------------------
    print("\n" + "=" * 50)
    if errors:
        print(f"Completado con {len(errors)} error(s):")
        for fname, exc in errors:
            print(f"  ✗ {fname}: {exc}")
    else:
        print(f"✓ Completado: {len(raw_files)} archivo(s) procesados.")
    print(f"Resultados guardados en: {folder}")


if __name__ == '__main__':
    main()
