"""
Richardson-Lucy deconvolution for OPM data.

CPU-based implementation using scikit-image. No GPU required.

PSF:
----
Automatically generated using a 3D Gaussian approximation from microscope
parameters (pixel size, NA, wavelength).

If `psfmodels` is installed (pip install psfmodels), a more accurate
vectorial PSF is used automatically.

Typical usage:
--------------
    from postprocessing.deconvolution import deconvolve_volume, generate_psf, load_experimental_psf

    # Option A: experimental PSF (measured from fluorescent beads)
    psf = load_experimental_psf("beads_deskew_488_0_psf.tif")

    # Option B: theoretical PSF (Gaussian, or vectorial if psfmodels is installed)
    psf = generate_psf(pixel_size_um=0.127, wavelength_nm=488, NA=1.1)

    # Deconvolve a deskewed volume (Z, Y, X)
    deconvolved = deconvolve_volume(deskewed, psf=psf, n_iter=15)

Parámetros típicos para el sistema LSM:
-----------------------------------------
    pixel_size_um = 0.127
    tilt_deg      = 41.0
    NA            = 1.1   (objetivo)
    wavelength_nm = 488 o 561
    n_iter        = 10-20 (más = más nitidez, más ruido)
"""

import numpy as np
from skimage.restoration import richardson_lucy
from tifffile import imread as _tif_imread

# Intentar importar psfmodels para PSF vectorial más precisa
_PSFMODELS_AVAILABLE = False
try:
    import psfmodels as psfm
    _PSFMODELS_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# PSF experimental
# =============================================================================

def load_experimental_psf(psf_path):
    """
    Carga una PSF experimental guardada como TIFF float32 (output de postprocess_psf.py).

    La PSF se normaliza a suma = 1 si aún no lo está.

    Parameters
    ----------
    psf_path : str
        Ruta al TIFF de PSF (ZYX, float32).

    Returns
    -------
    psf : np.ndarray (Z, Y, X), float32
        PSF normalizada, suma = 1.
    """
    psf = _tif_imread(psf_path).astype(np.float32)
    if psf.ndim != 3:
        raise ValueError(f"La PSF debe ser 3D (ZYX), recibida shape={psf.shape}")
    psf -= psf.min()
    total = psf.sum()
    if total > 0:
        psf /= total
    print(f"[PSF] PSF experimental cargada: {psf_path}  shape={psf.shape}")
    return psf


# =============================================================================
# Generación de PSF teórica
# =============================================================================

def generate_psf_gaussian(pixel_size_um, wavelength_nm, NA=1.1,
                          tilt_deg=41.0, n_xy=21, n_z=11):
    """
    PSF 3D Gaussiana aproximada en coordenadas de cámara OPM (skewed).

    Para el espacio skewed, el eje Z de la cámara corresponde al eje
    oblicuo → la extensión axial se proyecta sobre el eje Y de la cámara.

    Parameters
    ----------
    pixel_size_um : float
        Tamaño de píxel de la cámara en µm.
    wavelength_nm : float
        Longitud de onda de emisión en nm.
    NA : float
        Apertura numérica del objetivo.
    tilt_deg : float
        Ángulo OPM en grados (afecta la extensión Z proyectada).
    n_xy : int
        Radio de la PSF en XY en píxeles (tamaño final: 2*n_xy+1).
    n_z : int
        Radio de la PSF en Z en píxeles (tamaño final: 2*n_z+1).

    Returns
    -------
    psf : ndarray (2*n_z+1, 2*n_xy+1, 2*n_xy+1), float32
        PSF normalizada a suma = 1.
    """
    wl_um = wavelength_nm / 1000.0

    # Resoluciones de difracción (criterio de Rayleigh → σ Gaussiana)
    xy_sigma_um = 0.42 * wl_um / NA
    z_sigma_um  = 1.77 * wl_um / NA ** 2

    # En píxeles
    xy_sigma_px = xy_sigma_um / pixel_size_um

    # Z en coordenadas de cámara OPM (proyección del eje axial)
    tilt_rad   = np.deg2rad(tilt_deg)
    z_sigma_px = z_sigma_um / (pixel_size_um / np.sin(tilt_rad))

    # Grid 3D centrado
    Z, Y, X = np.mgrid[-n_z:n_z+1, -n_xy:n_xy+1, -n_xy:n_xy+1]

    psf = np.exp(
        -(X**2 + Y**2) / (2.0 * xy_sigma_px**2)
        - Z**2          / (2.0 * z_sigma_px**2)
    ).astype(np.float32)

    psf /= psf.sum()
    return psf


def generate_psf_vectorial(pixel_size_um, wavelength_nm, NA=1.1,
                            ni=1.33, tilt_deg=41.0, n_xy=21, nz_planes=15):
    """
    PSF vectorial 3D usando psfmodels (más precisa que la Gaussiana).

    Requiere: pip install psfmodels

    La PSF se genera en coordenadas de cubreobjetos (coverslip) y luego
    se mapea a coordenadas de cámara OPM mediante rotación.

    Parameters
    ----------
    pixel_size_um : float
    wavelength_nm : float
    NA : float
    ni : float
        Índice de refracción del medio de inmersión (agua=1.33, silicona=1.4).
    tilt_deg : float
        Ángulo OPM en grados.
    n_xy : int
        Radio de la PSF en XY (tamaño: 2*n_xy+1 píxeles).
    nz_planes : int
        Número de planos Z para la PSF.

    Returns
    -------
    psf : ndarray (nz_planes, 2*n_xy+1, 2*n_xy+1), float32
        PSF normalizada.
    """
    if not _PSFMODELS_AVAILABLE:
        print("[PSF] psfmodels no disponible. Usando PSF Gaussiana.")
        return generate_psf_gaussian(pixel_size_um, wavelength_nm, NA, tilt_deg,
                                     n_xy, nz_planes // 2)

    wl_um     = wavelength_nm / 1000.0
    tilt_rad  = np.deg2rad(tilt_deg)

    # Resolución axial estimada para dimensionar el volumen de la PSF
    z_sigma_um = 1.77 * wl_um / NA**2
    dz_um      = pixel_size_um / np.sin(tilt_rad)  # paso Z en el coverslip

    params = {
        'ni0': ni, 'ni': ni,
        'tg0': 170, 'tg': 170,
        'ns': ni,
        'ti0': 300,
        'NA': NA,
    }

    nxy  = 2 * n_xy + 1
    lim  = (nz_planes - 1) * dz_um / 2.0
    zv   = np.linspace(-lim, lim, nz_planes)

    try:
        psf = psfm.vectorial_psf(
            zv=zv, nx=nxy, dxy=pixel_size_um,
            pz=0.0, wvl=wl_um, params=params,
        )
        psf = psf / psf.sum()
    except Exception as e:
        print(f"[PSF] Error generando PSF vectorial: {e}. Usando Gaussiana.")
        return generate_psf_gaussian(pixel_size_um, wavelength_nm, NA, tilt_deg,
                                     n_xy, nz_planes // 2)

    return psf.astype(np.float32)


def generate_psf(pixel_size_um=0.127, wavelength_nm=488, NA=1.1,
                 tilt_deg=41.0, vectorial=True):
    """
    Genera la PSF teórica OPM. Usa psfmodels si está disponible.

    Parameters
    ----------
    pixel_size_um : float
    wavelength_nm : float
    NA : float
    tilt_deg : float
    vectorial : bool
        Si True, intenta usar psfmodels (vectorial). Si False o no disponible,
        usa aproximación Gaussiana.

    Returns
    -------
    psf : ndarray (Z, Y, X), float32
    """
    if vectorial and _PSFMODELS_AVAILABLE:
        print(f"[PSF] Generando PSF vectorial (λ={wavelength_nm}nm, NA={NA})")
        return generate_psf_vectorial(pixel_size_um, wavelength_nm, NA,
                                      tilt_deg=tilt_deg)
    else:
        mode = "Gaussiana (psfmodels no instalado)" if not _PSFMODELS_AVAILABLE else "Gaussiana"
        print(f"[PSF] Generando PSF {mode} (λ={wavelength_nm}nm, NA={NA})")
        return generate_psf_gaussian(pixel_size_um, wavelength_nm, NA, tilt_deg)


# =============================================================================
# Deconvolución
# =============================================================================

def deconvolve_volume(image, psf=None, n_iter=15,
                      pixel_size_um=0.127, tilt_deg=41.0,
                      wavelength_nm=488, NA=1.1):
    """
    Aplica deconvolución Richardson-Lucy a un volumen 3D OPM.

    La deconvolución se aplica en el espacio donde está el volumen
    (puede ser crudo/skewed o deskewed; en ambos casos funciona).

    Parameters
    ----------
    image : ndarray (Z, Y, X), uint16
        Volumen de entrada.
    psf : ndarray (Z, Y, X), optional
        PSF teórica. Si None, se genera automáticamente.
    n_iter : int
        Iteraciones Richardson-Lucy.
        - 5-10: suavizado leve, bajo riesgo de artefactos
        - 15-25: buena restauración, ruido tolerable
        - >30: puede amplificar ruido (usar solo con muy buen S/N)
    pixel_size_um : float
    tilt_deg : float
    wavelength_nm : float
    NA : float

    Returns
    -------
    deconvolved : ndarray (Z, Y, X), uint16
    """
    if image.ndim != 3:
        raise ValueError(f"Esperado 3D (Z,Y,X), recibido shape={image.shape}")

    if psf is None:
        psf = generate_psf(pixel_size_um, wavelength_nm, NA, tilt_deg)

    print(f"[Deconv] Input: {image.shape}, dtype={image.dtype}")
    print(f"[Deconv] PSF:   {psf.shape}")
    print(f"[Deconv] Iteraciones: {n_iter} ...")

    img_max = float(image.max())
    if img_max == 0:
        print("[Deconv] ADVERTENCIA: imagen toda en cero. Devolviendo sin cambios.")
        return image.copy()

    # Normalizar a [0, 1] para RL (trabaja mejor así)
    img_f = image.astype(np.float64) / img_max

    # PSF normalizada a suma = 1 (requerido por skimage)
    psf_norm = psf.astype(np.float64)
    psf_norm /= psf_norm.sum()

    result = richardson_lucy(img_f, psf_norm, num_iter=n_iter, clip=False)

    # Volver a escala original uint16
    result = np.clip(result * img_max, 0, 65535).astype(np.uint16)

    print(f"[Deconv] Listo. Output: {result.shape}")
    return result


def deconvolve_stack(stack, psf=None, n_iter=15, **kwargs):
    """
    Aplica deconvolución a cada frame de un timelapse (T, Z, Y, X).

    Parameters
    ----------
    stack : ndarray (T, Z, Y, X) o (Z, Y, X)
    psf : ndarray, optional
    n_iter : int
    **kwargs : pasados a deconvolve_volume()

    Returns
    -------
    result : mismo shape que stack
    """
    if stack.ndim == 3:
        return deconvolve_volume(stack, psf=psf, n_iter=n_iter, **kwargs)

    # Generar la PSF una sola vez para todos los frames
    if psf is None:
        psf = generate_psf(**{k: v for k, v in kwargs.items()
                               if k in ('pixel_size_um', 'wavelength_nm',
                                        'NA', 'tilt_deg')})

    result = np.zeros_like(stack)
    for t in range(stack.shape[0]):
        print(f"\n[Deconv] Frame {t+1}/{stack.shape[0]}")
        result[t] = deconvolve_volume(stack[t], psf=psf, n_iter=n_iter, **kwargs)
    return result


# =============================================================================
# Main / uso standalone
# =============================================================================

if __name__ == '__main__':
    import sys
    import argparse
    import tifffile

    parser = argparse.ArgumentParser(
        description="Deconvolución Richardson-Lucy para datos OPM")
    parser.add_argument('input', help='Archivo TIFF de entrada (Z, Y, X)')
    parser.add_argument('output', help='Archivo TIFF de salida')
    parser.add_argument('--n-iter',      type=int,   default=15)
    parser.add_argument('--wavelength',  type=float, default=488,
                        help='Longitud de onda de emisión en nm')
    parser.add_argument('--NA',          type=float, default=1.1)
    parser.add_argument('--pixel-size',  type=float, default=0.127,
                        help='Tamaño de píxel en µm')
    parser.add_argument('--tilt',        type=float, default=41.0,
                        help='Ángulo OPM en grados')
    parser.add_argument('--no-vectorial', action='store_true',
                        help='Usar PSF Gaussiana aunque psfmodels esté disponible')
    parser.add_argument('--psf-path', type=str, default=None,
                        help='TIFF de PSF experimental (output de postprocess_psf.py). '
                             'Si se provee, ignora --wavelength / --NA / --no-vectorial.')

    args = parser.parse_args()

    print(f"[Deconv] Leyendo: {args.input}")
    data = tifffile.imread(args.input)

    if args.psf_path:
        psf = load_experimental_psf(args.psf_path)
    else:
        psf = generate_psf(
            pixel_size_um=args.pixel_size,
            wavelength_nm=args.wavelength,
            NA=args.NA,
            tilt_deg=args.tilt,
            vectorial=not args.no_vectorial,
        )

    result = deconvolve_volume(
        data, psf=psf, n_iter=args.n_iter,
        pixel_size_um=args.pixel_size,
        tilt_deg=args.tilt,
        wavelength_nm=args.wavelength,
        NA=args.NA,
    )

    tifffile.imwrite(args.output, result, imagej=True, metadata={'axes': 'ZYX'})
    print(f"[Deconv] Guardado: {args.output}")
