"""
Flatfield correction para datos OPM.

Estima la no-uniformidad de iluminación a partir de un stack de imágenes crudas
y aplica la corrección. No requiere GPU ni dependencias especiales.

Algoritmo:
----------
1. Muestrea N frames aleatorios del stack.
2. Calcula la MEDIANA por pixel (suprime la señal de la muestra).
3. Suaviza el resultado con un filtro Gaussiano (iluminación lenta en espacio).
4. Normaliza a [0.01, 1.0] para evitar división por cero.
5. Corrige: corrected = (image - offset) / flatfield

Uso típico:
-----------
    from postprocess_flatfield import estimate_flatfield, apply_flatfield

    # Estimar desde stack crudo (Z, Y, X)
    ff = estimate_flatfield(raw_stack, camera_offset=100)

    # Aplicar a cada volumen
    corrected = apply_flatfield(volume, ff, camera_offset=100)
"""

import os
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter


# =============================================================================
# Estimación del flatfield
# =============================================================================

def estimate_flatfield(data_stack, smooth_sigma=50, n_samples=100,
                       camera_offset=100):
    """
    Estima el perfil de iluminación a partir de un stack de imágenes crudas.

    Parameters
    ----------
    data_stack : ndarray (Z, Y, X) o (T, Z, Y, X)
        Stack de imágenes crudas (sin deskew, sin corrección previa).
    smooth_sigma : float
        Sigma del suavizado Gaussiano en píxeles.
        Valores más grandes = flatfield más uniforme (rango típico: 30-80).
    n_samples : int
        Cuántos frames usar para la estimación. Más = mejor estimación,
        pero más lento. Típico: 50-200.
    camera_offset : int
        Offset de fondo de la cámara en ADU. Para PCO.edge 4.2: ~100.

    Returns
    -------
    flatfield : ndarray (Y, X), float32
        Perfil de iluminación normalizado entre 0.01 y 1.0.
        Dividir la imagen corregida por este mapa.

    Notes
    -----
    - El uso de la MEDIANA (en vez de la media) suprime la señal
      de la muestra y extrae solo la iluminación de fondo.
    - smooth_sigma grande (50+) asume que la iluminación varía
      lentamente. Reducir si hay patrones de alta frecuencia.
    """
    # Aplanar a (N, Y, X)
    flat = data_stack.reshape(-1, data_stack.shape[-2], data_stack.shape[-1])
    n_frames = flat.shape[0]

    # Seleccionar frames aleatorios
    n_use = min(n_samples, n_frames)
    idx   = np.random.choice(n_frames, size=n_use, replace=False)

    sampled = flat[idx].astype(np.float32) - camera_offset
    sampled = np.clip(sampled, 0.0, None)

    # Mediana: suprime estructura de la muestra
    illum = np.median(sampled, axis=0)

    # Suavizado Gaussiano: solo queda la variación lenta de iluminación
    illum_smooth = gaussian_filter(illum, sigma=smooth_sigma)

    # Normalizar
    max_val = illum_smooth.max()
    if max_val > 0:
        illum_smooth = illum_smooth / max_val

    # Clipear mínimo para evitar división por cero
    illum_smooth = np.clip(illum_smooth, 0.01, 1.0)

    return illum_smooth.astype(np.float32)


def estimate_flatfield_from_tiffs(tiff_paths, smooth_sigma=50, n_files=30,
                                  camera_offset=100):
    """
    Estima el flatfield a partir de una lista de archivos TIFF.

    Útil cuando los datos ya están guardados en disco y se quiere
    estimar el flatfield sin cargar todo en memoria.

    Parameters
    ----------
    tiff_paths : list of str
        Lista de rutas a archivos TIFF (raw o crop, sin deskew).
    smooth_sigma : float
        Ver estimate_flatfield().
    n_files : int
        Cuántos archivos usar (se samplea aleatoriamente si hay más).
    camera_offset : int
        Offset de la cámara.

    Returns
    -------
    flatfield : ndarray (Y, X), float32
    """
    import random

    sampled_paths = random.sample(tiff_paths, min(n_files, len(tiff_paths)))

    images = []
    for p in sampled_paths:
        img = tifffile.imread(p).astype(np.float32)
        # Si es 3D, usar la mitad del stack
        if img.ndim == 3:
            img = img[img.shape[0] // 2]
        images.append(img - camera_offset)

    images = np.stack(images, axis=0)
    images = np.clip(images, 0.0, None)

    illum        = np.median(images, axis=0)
    illum_smooth = gaussian_filter(illum, sigma=smooth_sigma)

    max_val = illum_smooth.max()
    if max_val > 0:
        illum_smooth = illum_smooth / max_val
    illum_smooth = np.clip(illum_smooth, 0.01, 1.0)

    return illum_smooth.astype(np.float32)


# =============================================================================
# Aplicación de la corrección
# =============================================================================

def apply_flatfield(image, flatfield, camera_offset=100):
    """
    Aplica la corrección de flatfield a una imagen o stack.

    Fórmula: corrected = (image - offset) / flatfield

    Parameters
    ----------
    image : ndarray (Y, X) o (Z, Y, X)
        Imagen cruda o stack a corregir.
    flatfield : ndarray (Y, X)
        Perfil de iluminación (normalizado a 1.0 en el máximo).
    camera_offset : int
        Offset de la cámara en ADU.

    Returns
    -------
    corrected : ndarray, uint16
        Imagen corregida con el mismo shape que la entrada.
    """
    img_f = image.astype(np.float32) - camera_offset
    img_f = np.clip(img_f, 0.0, None)

    if image.ndim == 3:
        corrected = img_f / flatfield[np.newaxis, :, :]
    else:
        corrected = img_f / flatfield

    return np.clip(corrected, 0, 65535).astype(np.uint16)


# =============================================================================
# I/O del flatfield
# =============================================================================

def save_flatfield(flatfield, path):
    """Guarda el flatfield en un TIFF de 32 bits."""
    tifffile.imwrite(path, flatfield)
    print(f"[Flatfield] Guardado: {path}")


def load_flatfield(path):
    """Carga un flatfield previamente guardado."""
    ff = tifffile.imread(path).astype(np.float32)
    # Validar que está normalizado
    if ff.max() > 1.1:
        print("[Flatfield] ADVERTENCIA: flatfield cargado parece no estar "
              "normalizado (max > 1.1). Normalizando...")
        ff = ff / ff.max()
        ff = np.clip(ff, 0.01, 1.0)
    return ff


# =============================================================================
# Uso standalone (procesar un directorio)
# =============================================================================

def process_directory(input_dir, output_dir=None, pattern='data_crop_*.tif',
                      smooth_sigma=50, n_files=30, camera_offset=100):
    """
    Estima y aplica flatfield a todos los TIFF de un directorio.

    Parameters
    ----------
    input_dir : str
        Directorio con archivos TIFF de entrada.
    output_dir : str, optional
        Directorio de salida (default: subdirectorio 'flatcorrected').
    pattern : str
        Glob pattern para filtrar archivos.
    smooth_sigma, n_files, camera_offset : ver estimate_flatfield_from_tiffs()

    Returns
    -------
    flatfield : ndarray (Y, X)
        Flatfield estimado y guardado en disco.
    """
    import glob

    if output_dir is None:
        output_dir = os.path.join(input_dir, 'flatcorrected')
    os.makedirs(output_dir, exist_ok=True)

    tiff_paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not tiff_paths:
        raise FileNotFoundError(f"No se encontraron archivos con patrón "
                                f"'{pattern}' en {input_dir}")

    print(f"[Flatfield] {len(tiff_paths)} archivos encontrados.")

    # Estimar flatfield
    print("[Flatfield] Estimando perfil de iluminación...")
    flatfield = estimate_flatfield_from_tiffs(
        tiff_paths, smooth_sigma=smooth_sigma,
        n_files=n_files, camera_offset=camera_offset,
    )

    ff_path = os.path.join(output_dir, 'flatfield.tif')
    save_flatfield(flatfield, ff_path)

    # Aplicar a cada archivo
    for i, path in enumerate(tiff_paths):
        fname   = os.path.basename(path)
        outpath = os.path.join(output_dir, fname)
        data    = tifffile.imread(path)
        corrected = apply_flatfield(data, flatfield, camera_offset=camera_offset)
        tifffile.imwrite(outpath, corrected, imagej=True, metadata={'axes': 'ZYX'})
        print(f"[Flatfield] [{i+1}/{len(tiff_paths)}] {fname}")

    print(f"[Flatfield] Listo. Output en: {output_dir}")
    return flatfield


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Uso: python postprocess_flatfield.py <directorio_datos>")
        print("     [--sigma 50] [--n-files 30] [--offset 100]")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(
        description="Flatfield correction para datos OPM")
    parser.add_argument('input_dir', help='Directorio con TIFFs de entrada')
    parser.add_argument('--output-dir', '-o', default=None)
    parser.add_argument('--pattern', default='data_crop_*.tif')
    parser.add_argument('--sigma', type=float, default=50.0)
    parser.add_argument('--n-files', type=int, default=30)
    parser.add_argument('--offset', type=int, default=100)

    args = parser.parse_args()
    process_directory(
        args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        smooth_sigma=args.sigma,
        n_files=args.n_files,
        camera_offset=args.offset,
    )
