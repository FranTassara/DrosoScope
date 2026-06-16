"""
Deskew para Oblique Plane Microscopy (OPM)
Basado en Shepherd Lab / QI2lab

Mejoras sobre la versión anterior:
- Z-downsampling por PROMEDIO (antes saltaba planos, ahora los integra → mejor S/N)
- Padding automático a múltiplos de 4 (mejor alineación de caché)
- Chequeos de bordes más estrictos (elimina artefactos en bordes)
- Recorte de bordes vacíos opcional (crop_after_deskew)
- Salida uint16 por defecto (antes era uint8, perdía rango dinámico)
- Proyección máxima Z integrada
"""

import numpy as np
from numba import njit, prange
import tifffile as tif
import os


# =============================================================================
# Núcleo JIT
# =============================================================================

@njit(parallel=True, cache=True)
def _deskew_core(data, theta_deg, distance_px, z_downsample=1, crop_y=0):
    """
    Núcleo deskew con Numba: interpolación ortogonal con promedio en Z.

    Parameters
    ----------
    data : ndarray (Z, Y, X), float32
        Stack OPM ya convertido a float32.
    theta_deg : float
        Ángulo del plano oblicuo en grados.
    distance_px : float
        Distancia entre planos en píxeles de cámara.
    z_downsample : int
        Factor de downsampling en Z. Los planos se PROMEDIAN (no se saltan).
    crop_y : int
        Píxeles a recortar de cada extremo en Y (elimina triángulos vacíos).

    Returns
    -------
    output : ndarray (Z', Y', X), float32
    """
    num_images, ny, nx = data.shape

    theta_rad = theta_deg * np.pi / 180.0
    tantheta  = np.float32(np.tan(theta_rad))
    sintheta  = np.float32(np.sin(theta_rad))
    costheta  = np.float32(np.cos(theta_rad))
    inv_dist  = np.float32(1.0 / distance_px)
    dist_f    = np.float32(distance_px)

    scan_end      = np.float32(num_images) * dist_f
    final_ny_raw  = np.int64(np.ceil(scan_end + np.float32(ny) * costheta))
    final_nz_full = np.int64(np.ceil(np.float32(ny) * sintheta))
    final_nz      = max(np.int64(1), final_nz_full // np.int64(z_downsample))

    # Y de salida con recorte
    final_ny = final_ny_raw - np.int64(2) * np.int64(crop_y)
    if final_ny < np.int64(1):
        final_ny = final_ny_raw  # fallback: sin recorte

    # Padding a múltiplos de 4 para mejor rendimiento de caché
    pad_y  = (np.int64(4) - (final_ny % np.int64(4))) % np.int64(4)
    pad_x  = (np.int64(4) - (np.int64(nx) % np.int64(4))) % np.int64(4)
    out_ny = final_ny + pad_y
    out_nx = np.int64(nx) + pad_x

    output = np.zeros((final_nz, out_ny, out_nx), dtype=np.float32)

    # -------------------------------------------------------------------------
    # Loop paralelo sobre slices Z reducidas (una por downsampled z)
    # -------------------------------------------------------------------------
    for z_ds in prange(final_nz):
        # Buffer acumulador por hilo (thread-private)
        buf   = np.zeros((final_ny_raw, nx), dtype=np.float32)
        count = np.int64(0)

        z_start = z_ds * np.int64(z_downsample)
        z_end   = min(z_start + np.int64(z_downsample), final_nz_full)

        # Acumular planos Z en el rango [z_start, z_end)
        for z in range(z_start, z_end):
            z_f = np.float32(z)

            for y in range(final_ny_raw):
                # Plano virtual de escaneo
                vplane = np.float32(y) - z_f / tantheta
                p0     = np.int64(np.floor(vplane * inv_dist))
                p1     = p0 + np.int64(1)

                # Chequeo de índice de plano
                if p0 < np.int64(0) or p1 >= np.int64(num_images):
                    continue

                lb = vplane - np.float32(p0) * dist_f  # fracción antes
                la = dist_f - lb                        # fracción después

                # Posición en el eje de la cámara (Y oblicuo)
                za = z_f / sintheta
                vb = za + lb * costheta   # posición antes
                va = za - la * costheta   # posición después

                pb = np.int64(np.floor(vb))
                pa = np.int64(np.floor(va))

                # Chequeo de índice de posición (estricto)
                if (pb < np.int64(0) or pa < np.int64(0) or
                        pb + np.int64(1) >= np.int64(ny) or
                        pa + np.int64(1) >= np.int64(ny)):
                    continue

                dzb = vb - np.float32(pb)
                dza = va - np.float32(pa)

                # Pesos de interpolación bilineal
                w1 = lb * dza         * inv_dist
                w2 = lb * (np.float32(1.0) - dza) * inv_dist
                w3 = la * dzb         * inv_dist
                w4 = la * (np.float32(1.0) - dzb) * inv_dist

                # Interpolación vectorial sobre X
                for xi in range(nx):
                    buf[y, xi] += (
                        w1 * data[p1, pa + np.int64(1), xi] +
                        w2 * data[p1, pa,               xi] +
                        w3 * data[p0, pb + np.int64(1), xi] +
                        w4 * data[p0, pb,               xi]
                    )

            count += np.int64(1)

        # Promediar y escribir a output (aplicando recorte de bordes)
        if count > np.int64(0):
            inv_count = np.float32(1.0) / np.float32(count)
            for y in range(final_ny):
                for xi in range(nx):
                    output[z_ds, y, xi] = buf[y + np.int64(crop_y), xi] * inv_count

    return output


# =============================================================================
# API pública
# =============================================================================

def deskew(data, theta, distance, pixel_size,
           z_downsample=1, output_dtype='uint16',
           crop_after_deskew=False):
    """
    Deskew de un stack OPM con interpolación ortogonal paralela.

    Parameters
    ----------
    data : ndarray (Z, Y, X)
        Stack de imágenes OPM. Puede ser uint8, uint16 o float.
    theta : float
        Ángulo del plano oblicuo relativo al coverslip en GRADOS.
    distance : float
        Distancia entre planos de imagen en MICRÓMETROS (µm).
    pixel_size : float
        Tamaño de píxel de la cámara en MICRÓMETROS (µm).
    z_downsample : int, optional
        Factor de downsampling en Z. Los planos se PROMEDIAN (no se saltan).
        Default=1 (sin downsampling). Usar 2 reduce Z a la mitad con mejor S/N.
    output_dtype : str, optional
        Tipo de dato de salida: 'uint8', 'uint16', o 'float32'.
        Default='uint16' (preserva el rango dinámico completo).
    crop_after_deskew : bool, optional
        Si True, recorta los triángulos vacíos de los bordes en Y.
        Útil cuando el rango de scan es mucho mayor que el FOV de la cámara.
        Default=False.

    Returns
    -------
    output : ndarray (Z', Y', X)
        Volumen deskewed en el dtype especificado.

    Notes
    -----
    - El primer run es lento porque Numba compila el núcleo JIT.
    - z_downsample > 1 PROMEDIA los planos (más S/N que el salto directo).
    - Default output_dtype cambió de 'uint8' a 'uint16' para preservar el
      rango dinámico. Cambiar a 'uint8' solo si el tamaño de archivo importa.

    Examples
    --------
    >>> deskewed = deskew(
    ...     data=raw_stack,
    ...     theta=41.0,
    ...     distance=0.505,
    ...     pixel_size=0.127,
    ...     z_downsample=2,
    ...     output_dtype='uint16',
    ... )
    """
    # Validación
    if data.ndim != 3:
        raise ValueError(f"data debe ser 3D (Z, Y, X), shape={data.shape}")
    if not (0 < theta < 90):
        raise ValueError(f"theta debe estar entre 0 y 90 grados, got {theta}")
    if distance <= 0:
        raise ValueError(f"distance debe ser > 0, got {distance}")
    if pixel_size <= 0:
        raise ValueError(f"pixel_size debe ser > 0, got {pixel_size}")
    if z_downsample < 1:
        raise ValueError(f"z_downsample debe ser >= 1, got {z_downsample}")
    if output_dtype not in ('uint8', 'uint16', 'float32'):
        raise ValueError(f"output_dtype inválido: '{output_dtype}'")

    data_f32    = data.astype(np.float32)
    distance_px = distance / pixel_size

    # Calcular recorte de bordes si se pide
    if crop_after_deskew:
        theta_rad = np.deg2rad(theta)
        crop_y    = int(np.ceil(data.shape[1] * np.cos(theta_rad)))
        # Verificar que el crop no borra todo
        final_ny_raw = int(np.ceil(
            data.shape[0] * distance_px + data.shape[1] * np.cos(theta_rad)
        ))
        if final_ny_raw - 2 * crop_y < 10:
            print("[Deskew] ADVERTENCIA: crop_after_deskew reduciría el output a "
                  f"{final_ny_raw - 2*crop_y} px. Desactivando crop.")
            crop_y = 0
    else:
        crop_y = 0

    print(f"[Deskew] Input:  {data.shape}, dtype={data.dtype}")
    print(f"[Deskew] theta={theta}°, distance={distance:.4f}µm, "
          f"pixel_size={pixel_size}µm, distance_px={distance_px:.3f}px")
    print(f"[Deskew] z_downsample={z_downsample} (promedio), "
          f"crop_y={crop_y}, dtype_out={output_dtype}")

    result = _deskew_core(data_f32, theta, distance_px, z_downsample, crop_y)

    print(f"[Deskew] Output: {result.shape}")

    # Conversión de dtype
    if output_dtype == 'float32':
        return result

    if output_dtype == 'uint16':
        return np.clip(result, 0, 65535).astype(np.uint16)

    # uint8: normalizar al rango 0-255
    rmin, rmax = result.min(), result.max()
    if rmax > rmin:
        result = (result - rmin) / (rmax - rmin) * 255.0
    return np.clip(result, 0, 255).astype(np.uint8)


def max_projection_z(volume):
    """
    Proyección de máxima intensidad a lo largo del eje Z.

    Parameters
    ----------
    volume : ndarray (Z, Y, X)
        Volumen 3D (puede ser deskewed o raw).

    Returns
    -------
    projection : ndarray (Y, X)
        Imagen 2D con el valor máximo en cada posición XY.
    """
    return np.max(volume, axis=0)


def deskew_and_save(input_path, output_path, theta, distance, pixel_size,
                    z_downsample=1, output_dtype='uint16',
                    crop_after_deskew=False, save_max_projection=True):
    """
    Conveniencia: lee, deskewea y guarda en un paso.

    Parameters
    ----------
    save_max_projection : bool
        Si True, guarda también la proyección máxima Z en _maxz.tif.

    Returns
    -------
    output_shape : tuple
    """
    print(f"[Deskew] Leyendo: {input_path}")
    data = tif.imread(input_path)

    result = deskew(data, theta, distance, pixel_size,
                    z_downsample, output_dtype, crop_after_deskew)

    print(f"[Deskew] Guardando: {output_path}")
    tif.imwrite(output_path, result, imagej=True, metadata={'axes': 'ZYX'})

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Deskew] Tamaño: {size_mb:.1f} MB")

    if save_max_projection:
        maxz       = max_projection_z(result)
        base, ext  = os.path.splitext(output_path)
        maxz_path  = base + '_maxz' + ext
        tif.imwrite(maxz_path, maxz, imagej=True, metadata={'axes': 'YX'})
        print(f"[Deskew] Max projection: {maxz_path}")

    return result.shape


# =============================================================================
# Estimación de tamaño (sin procesar)
# =============================================================================

def estimate_output_size(input_shape, theta, distance, pixel_size,
                         z_downsample=1, dtype='uint16',
                         crop_after_deskew=False):
    """
    Estima el tamaño del volumen deskewed sin procesarlo.

    Útil para verificar RAM/disco antes de procesar.
    """
    num_images, ny, nx = input_shape
    distance_px = distance / pixel_size
    theta_rad   = np.deg2rad(theta)

    scan_end      = num_images * distance_px
    final_ny_raw  = int(np.ceil(scan_end + ny * np.cos(theta_rad)))
    final_nz_full = int(np.ceil(ny * np.sin(theta_rad)))
    final_nz      = max(1, final_nz_full // z_downsample)

    if crop_after_deskew:
        crop_y   = int(np.ceil(ny * np.cos(theta_rad)))
        final_ny = max(1, final_ny_raw - 2 * crop_y)
    else:
        final_ny = final_ny_raw

    # Padding a múltiplos de 4
    pad_y    = (4 - (final_ny % 4)) % 4
    pad_x    = (4 - (nx % 4)) % 4
    final_ny = final_ny + pad_y
    final_nx = nx + pad_x

    bytes_per_pixel = {'uint8': 1, 'uint16': 2, 'float32': 4}[dtype]
    total_bytes     = final_nz * final_ny * final_nx * bytes_per_pixel

    return {
        'output_shape': (final_nz, final_ny, final_nx),
        'size_mb':  total_bytes / (1024 ** 2),
        'size_gb':  total_bytes / (1024 ** 3),
    }


# =============================================================================
# Helpers I/O
# =============================================================================

def imread(filename):
    """Lee TIFF manteniendo orden ZYX."""
    with tif.TiffFile(filename) as t:
        axes       = t.series[0].axes
        hyperstack = t.series[0].asarray()
    return tif.transpose_axes(hyperstack, axes, 'ZYX')


def imwrite(filename, data, dtype=None):
    """Guarda TIFF con metadata ZYX."""
    if dtype == 'uint8':
        data = np.clip(data, 0, 255).astype(np.uint8)
    elif dtype == 'uint16':
        data = np.clip(data, 0, 65535).astype(np.uint16)
    return tif.imwrite(filename, data, imagej=True, metadata={'axes': 'ZYX'})


# =============================================================================
# Main — ejemplo de uso
# =============================================================================

if __name__ == "__main__":
    input_directory = 'D:/2026/03.31_invivo_mitogfppdfrfp/31.03_mitogfppdfrfp_invivo_561/crop_v2'
    input_file      = 'data_cropv2_561_0.tif'

    theta       = 41.0
    pixel_size  = 0.127    # µm por píxel
    z_downsample = 2       # promedia pares de planos Z
    output_dtype = 'uint16'

    input_path  = os.path.join(input_directory, input_file)
    output_file = (f"{os.path.splitext(input_file)[0]}"
                   f"_deskewed_theta{theta}_z{z_downsample}_{output_dtype}.tif")
    output_path = os.path.join(input_directory, output_file)

    data = tif.imread(input_path)
    print(f"Shape entrada: {data.shape}, dtype: {data.dtype}")

    nz       = data.shape[0]
    scan_range_um = 60.0
    distance = scan_range_um / nz
    print(f"distance: {distance:.4f} µm/plano")

    # Preparar datos (transponer Y↔X y flipear dirección del scan)
    data_prep = np.flip(np.transpose(data, (0, 2, 1)), axis=0)

    est = estimate_output_size(data_prep.shape, theta, distance, pixel_size,
                               z_downsample, output_dtype)
    print(f"Shape estimada: {est['output_shape']}  "
          f"({est['size_mb']:.1f} MB / {est['size_gb']:.2f} GB)")

    result = deskew(data_prep, theta, distance, pixel_size,
                    z_downsample, output_dtype)

    imwrite(output_path, result)

    # Max projection
    maxz = max_projection_z(result)
    maxz_path = output_path.replace('.tif', '_maxz.tif')
    tif.imwrite(maxz_path, maxz, imagej=True, metadata={'axes': 'YX'})

    print(f"Listo! Output: {output_path}")
    print(f"Max Z: {maxz_path}")
