"""
Deconvolución batch de los volúmenes data_deskew_488_*.tif usando PSF experimental.

Pipeline por volumen:
  1. Cargar volumen (Z, Y, X)
  2. Recortar CROP_FRAMES frames del inicio y del final en Z
  3. Aplicar deconvolución Richardson-Lucy

Modos:
  PSF_MODE = "2D"  → PSF (Y, X):   Richardson-Lucy aplicado plano a plano (slice-by-slice).
  PSF_MODE = "3D"  → PSF (Z, Y, X): Richardson-Lucy aplicado sobre el volumen completo.

PSF   : D:/2026/03.31_invivo_mitogfppdfrfp/31.03_mitogfppdfrfp_invivo_488/
        data_deskewed/data_deskewed_deconvolved/psf_3D_488.tif
Input : D:/2026/03.31_invivo_mitogfppdfrfp/31.03_mitogfppdfrfp_invivo_488/
        data_deskewed/data_deskew_488_*.tif
Output: misma carpeta que la PSF (data_deskewed_deconvolved/)
"""

import sys
import os
import re
import time
import glob
import numpy as np
import tifffile
from skimage.restoration import richardson_lucy

# =============================================================================
# Configuración
# =============================================================================

PSF_PATH   = "D:/2026/03.31_invivo_mitogfppdfrfp/31.03_mitogfppdfrfp_invivo_488/data_deskewed/data_deskewed_deconvolved2D/psf_2D_488.tif"
INPUT_DIR  = "D:/2026/03.31_invivo_mitogfppdfrfp/31.03_mitogfppdfrfp_invivo_488/data_deskewed"
OUTPUT_DIR = "D:/2026/03.31_invivo_mitogfppdfrfp/31.03_mitogfppdfrfp_invivo_488/data_deskewed/data_deskewed_deconvolved2D"
N_ITER      = 10
# "2D" → deconvolución plano a plano con PSF (Y, X)
# "3D" → deconvolución volumétrica con PSF (Z, Y, X)
PSF_MODE    = "2D"
# Número de frames a eliminar al inicio Y al final del eje Z antes de deconvolucionar
CROP_FRAMES = 4

# =============================================================================
# Helpers
# =============================================================================

def natural_sort_key(path):
    name = os.path.basename(path)
    nums = re.findall(r'\d+', name)
    return int(nums[-1]) if nums else 0


def load_psf(psf_path, mode):
    psf = tifffile.imread(psf_path).astype(np.float64)
    psf -= psf.min()
    psf /= psf.sum()

    if mode == "2D" and psf.ndim != 2:
        raise ValueError(f"PSF_MODE='2D' pero la PSF tiene shape={psf.shape} (se esperaba 2D)")
    if mode == "3D" and psf.ndim != 3:
        raise ValueError(f"PSF_MODE='3D' pero la PSF tiene shape={psf.shape} (se esperaba 3D)")

    print(f"[PSF] Cargada ({mode}): {psf_path}  shape={psf.shape}")
    return psf


def deconvolve_2d(volume, psf_2d, n_iter):
    """Richardson-Lucy 2D aplicado a cada plano Z del volumen."""
    result = np.zeros_like(volume)
    img_max = float(volume.max())
    if img_max == 0:
        print("[Deconv] ADVERTENCIA: volumen todo en cero.")
        return volume.copy()

    for z in range(volume.shape[0]):
        plane = volume[z].astype(np.float64) / img_max
        deconv = richardson_lucy(plane, psf_2d, num_iter=n_iter, clip=False)
        result[z] = np.clip(deconv * img_max, 0, 65535).astype(np.uint16)

    return result


def deconvolve_3d(volume, psf_3d, n_iter):
    """Richardson-Lucy 3D aplicado sobre el volumen completo."""
    img_max = float(volume.max())
    if img_max == 0:
        print("[Deconv] ADVERTENCIA: volumen todo en cero.")
        return volume.copy()

    img_f = volume.astype(np.float64) / img_max
    result = richardson_lucy(img_f, psf_3d, num_iter=n_iter, clip=False)
    return np.clip(result * img_max, 0, 65535).astype(np.uint16)


# =============================================================================
# Main
# =============================================================================

def main():
    if PSF_MODE not in ("2D", "3D"):
        print(f"[ERROR] PSF_MODE debe ser '2D' o '3D', recibido: '{PSF_MODE}'")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    psf = load_psf(PSF_PATH, PSF_MODE)

    pattern = os.path.join(INPUT_DIR, "data_deskew_488_*.tif")
    files = sorted(glob.glob(pattern), key=natural_sort_key)

    if not files:
        print(f"[ERROR] No se encontraron archivos: {pattern}")
        sys.exit(1)

    mode_label = "2D slice-by-slice" if PSF_MODE == "2D" else "3D volumétrica"
    print(f"\n[Batch] {len(files)} volúmenes encontrados. n_iter={N_ITER}  modo={mode_label}\n")

    t_total_start = time.time()

    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        out_name = fname.replace("data_deskew_488_", "data_deskew_488_deconv_")
        out_path = os.path.join(OUTPUT_DIR, out_name)

        if os.path.exists(out_path):
            print(f"[{i+1}/{len(files)}] Ya existe, saltando: {out_name}")
            continue

        print(f"\n[{i+1}/{len(files)}] {fname}")
        t0 = time.time()

        data = tifffile.imread(fpath)
        print(f"  shape={data.shape}  dtype={data.dtype}")

        # Recortar frames al inicio y al final en Z
        if CROP_FRAMES > 0:
            data = data[CROP_FRAMES:-CROP_FRAMES]
            print(f"  recortado → shape={data.shape}  (±{CROP_FRAMES} frames eliminados)")

        if PSF_MODE == "2D":
            result = deconvolve_2d(data, psf, N_ITER)
        else:
            result = deconvolve_3d(data, psf, N_ITER)

        tifffile.imwrite(out_path, result, imagej=True, metadata={'axes': 'ZYX'})
        dt = time.time() - t0
        print(f"  Guardado: {out_name}  ({dt:.1f}s)")

    elapsed = time.time() - t_total_start
    print(f"\n[Batch] Terminado. Total: {elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
