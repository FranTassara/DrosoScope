"""
OPM Post-processing Pipeline
==============================

Post-processing chain for data acquired with DrosoScope:

    [Raw TIFF] → Flatfield correction → Deconvolution → Deskew → Max Projection

Each step is optional and can be combined freely.

Command-line usage:
-------------------
    # Minimum (deskew only):
    python -m postprocessing.pipeline data/data_crop_488_0.tif --scan-range 70

    # With flatfield correction:
    python -m postprocessing.pipeline data/data_crop_488_0.tif --scan-range 70 --flatfield

    # With deconvolution (488 nm):
    python -m postprocessing.pipeline data/data_crop_488_0.tif --scan-range 70 --deconvolve --wavelength 488

    # Full pipeline:
    python -m postprocessing.pipeline data/data_crop_488_0.tif --scan-range 70 \\
        --flatfield --deconvolve --wavelength 488 -o results/

    # Process all TIFFs in a directory:
    python -m postprocessing.pipeline data/ --scan-range 70 --flatfield --deconvolve

Programmatic usage:
-------------------
    from postprocessing.pipeline import postprocess

    postprocess(
        input_path       = "data/data_crop_488_0.tif",
        scan_range_um    = 70.0,
        do_flatfield     = True,
        do_deconvolution = True,
        wavelength_nm    = 488,
    )

Fixed system parameters:
-------------------------
    theta_deg     = 41.0   (OPM oblique angle)
    pixel_size_um = 0.127  (PCO.edge 4.2 with current objective)
    NA            = 1.1
    camera_offset = 100    (PCO camera background offset)
"""

import os
import glob
import argparse
import numpy as np
import tifffile

from .deskew import deskew, max_projection_z, estimate_output_size
from .flatfield import (estimate_flatfield, estimate_flatfield_from_tiffs,
                        apply_flatfield, save_flatfield, load_flatfield)
from .deconvolution import deconvolve_volume, generate_psf


# =============================================================================
# Pipeline principal
# =============================================================================

def postprocess(
    input_path,
    output_dir        = None,
    # Parámetros de adquisición
    theta_deg         = 41.0,
    pixel_size_um     = 0.127,
    scan_range_um     = None,
    distance_um       = None,
    # Flatfield
    do_flatfield       = False,
    flatfield_path     = None,   # ruta a flatfield.tif pre-calculado
    camera_offset      = 100,
    flatfield_sigma    = 50,
    flatfield_n_samples= 80,
    # Deconvolución
    do_deconvolution   = False,
    n_iter             = 15,
    wavelength_nm      = 488,
    NA                 = 1.1,
    # Deskew
    z_downsample       = 2,
    output_dtype       = 'uint16',
    crop_after_deskew  = False,
    # Salidas
    save_flatcorrected = False,   # guardar el volumen flat-corregido (crudo)
    save_deconvolved   = False,   # guardar el volumen deconvolucionado (crudo)
    save_deskewed      = True,
    save_max_projection= True,
):
    """
    Aplica la cadena de post-procesamiento a un archivo TIFF OPM.

    Parameters
    ----------
    input_path : str
        Ruta al archivo TIFF de entrada (Z, Y, X) — raw o crop, sin deskew.
    output_dir : str, optional
        Directorio de salida. Por defecto, junto al archivo de entrada.
    theta_deg : float
        Ángulo OPM en grados (default 41.0).
    pixel_size_um : float
        Tamaño de píxel de la cámara en µm (default 0.127).
    scan_range_um : float, optional
        Rango total del scan en µm. Se usa para calcular distance_um.
    distance_um : float, optional
        Paso entre planos en µm. Alternativa a scan_range_um.
    do_flatfield : bool
        Activar corrección de flatfield (default False).
    flatfield_path : str, optional
        Ruta a un flatfield.tif pre-calculado. Si None y do_flatfield=True,
        se estima desde los datos.
    camera_offset : int
        Offset de la cámara en ADU (default 100 para PCO.edge 4.2).
    flatfield_sigma : float
        Sigma del suavizado Gaussiano para estimar flatfield (default 50).
    do_deconvolution : bool
        Activar deconvolución Richardson-Lucy (default False).
    n_iter : int
        Iteraciones de RL (default 15). Rango típico: 10-25.
    wavelength_nm : float
        Longitud de onda de emisión en nm (default 488).
    NA : float
        Apertura numérica del objetivo (default 1.1).
    z_downsample : int
        Factor de downsampling en Z por promedio (default 2).
    output_dtype : str
        Dtype del deskew: 'uint16' o 'float32' (default 'uint16').
    save_max_projection : bool
        Guardar proyección máxima Z del deskew (default True).

    Returns
    -------
    result : dict con arrays resultantes (deskewed, maxz, etc.)
    """

    print(f"\n{'='*62}")
    print(f"  OPM Post-processing: {os.path.basename(input_path)}")
    print(f"{'='*62}")

    # --- Setup output ---
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_path))
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_path))[0]

    # --- Cargar datos ---
    print(f"\n[1] Cargando datos...")
    data = tifffile.imread(input_path)
    if data.ndim != 3:
        raise ValueError(f"Esperado 3D (Z, Y, X), recibido shape={data.shape}")
    print(f"    Shape: {data.shape}, dtype: {data.dtype}")

    # --- Calcular paso entre planos ---
    if distance_um is None:
        if scan_range_um is None:
            raise ValueError("Proveer scan_range_um o distance_um.")
        distance_um = scan_range_um / data.shape[0]
    print(f"    distance_um = {distance_um:.5f} µm/plano "
          f"(scan total ≈ {distance_um * data.shape[0]:.1f} µm)")

    # Estimar tamaño del output para referencia
    est = estimate_output_size(
        (data.shape[0], data.shape[2], data.shape[1]),  # transpuesto como en la app
        theta=theta_deg, distance=distance_um,
        pixel_size=pixel_size_um, z_downsample=z_downsample,
        dtype=output_dtype,
    )
    print(f"    Deskew estimado: {est['output_shape']}  "
          f"({est['size_mb']:.0f} MB)")

    current = data.copy()
    outputs = {}

    # =========================================================================
    # PASO 1: Flatfield correction
    # =========================================================================
    step = 2
    if do_flatfield:
        print(f"\n[{step}] Flatfield correction...")
        step += 1

        if flatfield_path and os.path.exists(flatfield_path):
            print(f"    Cargando flatfield: {flatfield_path}")
            flatfield = load_flatfield(flatfield_path)
        else:
            print(f"    Estimando flatfield desde los datos "
                  f"(sigma={flatfield_sigma}, n={flatfield_n_samples})...")
            flatfield = estimate_flatfield(
                data,
                smooth_sigma=flatfield_sigma,
                n_samples=flatfield_n_samples,
                camera_offset=camera_offset,
            )
            ff_save = os.path.join(output_dir, f'{base}_flatfield.tif')
            save_flatfield(flatfield, ff_save)
            outputs['flatfield'] = flatfield

        current = apply_flatfield(current, flatfield, camera_offset=camera_offset)
        print(f"    Corrección aplicada.")

        if save_flatcorrected:
            fc_path = os.path.join(output_dir, f'{base}_flatcorrected.tif')
            tifffile.imwrite(fc_path, current, imagej=True,
                             metadata={'axes': 'ZYX'})
            print(f"    Guardado: {fc_path}")

        outputs['flat_corrected'] = current
    else:
        print(f"\n[{step}] Flatfield: omitido")
        step += 1

    # =========================================================================
    # PASO 2: Deconvolución
    # =========================================================================
    if do_deconvolution:
        print(f"\n[{step}] Deconvolución RL (λ={wavelength_nm}nm, "
              f"{n_iter} iter, NA={NA})...")
        step += 1

        psf = generate_psf(
            pixel_size_um=pixel_size_um,
            wavelength_nm=wavelength_nm,
            NA=NA,
            tilt_deg=theta_deg,
        )

        current = deconvolve_volume(
            current, psf=psf, n_iter=n_iter,
            pixel_size_um=pixel_size_um,
            tilt_deg=theta_deg,
            wavelength_nm=wavelength_nm,
            NA=NA,
        )

        if save_deconvolved:
            dc_path = os.path.join(output_dir, f'{base}_deconvolved.tif')
            tifffile.imwrite(dc_path, current, imagej=True,
                             metadata={'axes': 'ZYX'})
            print(f"    Guardado: {dc_path}")

        outputs['deconvolved'] = current
    else:
        print(f"\n[{step}] Deconvolución: omitida")
        step += 1

    # =========================================================================
    # PASO 3: Deskew
    # =========================================================================
    print(f"\n[{step}] Deskew (z_downsample={z_downsample}, "
          f"dtype={output_dtype})...")
    step += 1

    # Preparar datos: transponer Y↔X y flipear dirección del scan
    # (misma operación que en App_mejorada.py)
    data_prep = np.flip(np.transpose(current, (0, 2, 1)), axis=0)

    deskewed = deskew(
        data      = data_prep,
        theta     = theta_deg,
        distance  = distance_um,
        pixel_size= pixel_size_um,
        z_downsample    = z_downsample,
        output_dtype    = output_dtype,
        crop_after_deskew = crop_after_deskew,
    )

    if save_deskewed:
        dk_path = os.path.join(output_dir, f'{base}_deskewed.tif')
        tifffile.imwrite(dk_path, deskewed, imagej=True,
                         metadata={'axes': 'ZYX'})
        print(f"    Guardado: {dk_path}")

    outputs['deskewed'] = deskewed

    # =========================================================================
    # PASO 4: Max projection Z
    # =========================================================================
    if save_max_projection:
        print(f"\n[{step}] Max projection Z...")
        maxz     = max_projection_z(deskewed)
        mz_path  = os.path.join(output_dir, f'{base}_maxz.tif')
        tifffile.imwrite(mz_path, maxz, imagej=True, metadata={'axes': 'YX'})
        print(f"    Guardado: {mz_path}")
        outputs['maxz'] = maxz

    print(f"\n{'='*62}")
    print(f"  Post-processing completado.")
    print(f"  Output: {output_dir}")
    print(f"{'='*62}\n")

    return outputs


# =============================================================================
# Batch: procesar un directorio
# =============================================================================

def postprocess_directory(input_dir, pattern='data_crop_*.tif',
                          output_dir=None, **kwargs):
    """
    Aplica el pipeline a todos los TIFFs que coinciden con un patrón.

    Parameters
    ----------
    input_dir : str
        Directorio que contiene los archivos TIFF.
    pattern : str
        Glob pattern para filtrar archivos (default 'data_crop_*.tif').
    output_dir : str, optional
        Directorio de salida. Por defecto, subdirectorio 'postprocessed'.
    **kwargs : pasados a postprocess()

    Returns
    -------
    results : list of dict
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'postprocessed')

    tiff_paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not tiff_paths:
        print(f"No se encontraron archivos con patrón '{pattern}' en {input_dir}")
        return []

    print(f"Encontrados {len(tiff_paths)} archivos.")

    # Si se pide flatfield, estimarlo una sola vez para todo el directorio
    if kwargs.get('do_flatfield') and not kwargs.get('flatfield_path'):
        print("\nEstimando flatfield del directorio (una sola vez)...")
        flatfield = estimate_flatfield_from_tiffs(
            tiff_paths,
            smooth_sigma=kwargs.get('flatfield_sigma', 50),
            n_files=min(30, len(tiff_paths)),
            camera_offset=kwargs.get('camera_offset', 100),
        )
        ff_path = os.path.join(output_dir, 'flatfield_batch.tif')
        os.makedirs(output_dir, exist_ok=True)
        save_flatfield(flatfield, ff_path)
        kwargs['flatfield_path'] = ff_path
        print(f"Flatfield guardado en: {ff_path}\n")

    results = []
    for i, path in enumerate(tiff_paths):
        print(f"\n{'─'*62}")
        print(f"Archivo {i+1}/{len(tiff_paths)}: {os.path.basename(path)}")
        result = postprocess(path, output_dir=output_dir, **kwargs)
        results.append(result)

    return results


# =============================================================================
# CLI
# =============================================================================

def _build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument('input',
                   help='Archivo TIFF de entrada o directorio (batch)')

    p.add_argument('--output-dir', '-o', default=None,
                   help='Directorio de salida')

    # Parámetros de adquisición
    acq = p.add_argument_group('Parámetros de adquisición')
    acq.add_argument('--theta',       type=float, default=41.0,
                     help='Ángulo OPM en grados (default 41.0)')
    acq.add_argument('--pixel-size',  type=float, default=0.127,
                     help='Tamaño de píxel en µm (default 0.127)')
    acq.add_argument('--scan-range',  type=float, default=None,
                     help='Rango total de scan en µm')
    acq.add_argument('--distance',    type=float, default=None,
                     help='Paso entre planos en µm')

    # Flatfield
    ff = p.add_argument_group('Flatfield correction')
    ff.add_argument('--flatfield', action='store_true',
                    help='Activar flatfield correction')
    ff.add_argument('--flatfield-file', default=None,
                    help='Ruta a flatfield.tif pre-calculado')
    ff.add_argument('--camera-offset', type=int, default=100,
                    help='Offset de cámara en ADU (default 100)')
    ff.add_argument('--ff-sigma', type=float, default=50.0,
                    help='Sigma Gaussiano para estimar flatfield (default 50)')

    # Deconvolución
    dc = p.add_argument_group('Deconvolución')
    dc.add_argument('--deconvolve', action='store_true',
                    help='Activar deconvolución Richardson-Lucy')
    dc.add_argument('--n-iter',     type=int,   default=15,
                    help='Iteraciones RL (default 15)')
    dc.add_argument('--wavelength', type=float, default=488,
                    help='Longitud de onda de emisión en nm (default 488)')
    dc.add_argument('--NA',         type=float, default=1.1,
                    help='Apertura numérica (default 1.1)')

    # Deskew
    dk = p.add_argument_group('Deskew')
    dk.add_argument('--z-downsample', type=int, default=2,
                    help='Factor de downsampling Z (default 2, promedio)')
    dk.add_argument('--dtype',  default='uint16',
                    choices=['uint16', 'float32'],
                    help='Dtype de salida del deskew (default uint16)')
    dk.add_argument('--crop',   action='store_true',
                    help='Recortar triángulos vacíos de bordes tras el deskew')

    # Salidas
    out = p.add_argument_group('Archivos de salida')
    out.add_argument('--save-flatcorrected', action='store_true',
                     help='Guardar el volumen corregido (antes de deskew)')
    out.add_argument('--save-deconvolved', action='store_true',
                     help='Guardar el volumen deconvolucionado (antes de deskew)')
    out.add_argument('--no-max-projection', action='store_true',
                     help='No guardar proyección máxima Z')

    # Batch
    p.add_argument('--pattern', default='data_crop_*.tif',
                   help='Glob pattern para procesamiento batch (default: data_crop_*.tif)')

    return p


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    common_kwargs = dict(
        output_dir         = args.output_dir,
        theta_deg          = args.theta,
        pixel_size_um      = args.pixel_size,
        scan_range_um      = args.scan_range,
        distance_um        = args.distance,
        do_flatfield       = args.flatfield,
        flatfield_path     = args.flatfield_file,
        camera_offset      = args.camera_offset,
        flatfield_sigma    = args.ff_sigma,
        do_deconvolution   = args.deconvolve,
        n_iter             = args.n_iter,
        wavelength_nm      = args.wavelength,
        NA                 = args.NA,
        z_downsample       = args.z_downsample,
        output_dtype       = args.dtype,
        crop_after_deskew  = args.crop,
        save_flatcorrected = args.save_flatcorrected,
        save_deconvolved   = args.save_deconvolved,
        save_max_projection= not args.no_max_projection,
    )

    if os.path.isdir(args.input):
        postprocess_directory(args.input, pattern=args.pattern, **common_kwargs)
    else:
        postprocess(args.input, **common_kwargs)


if __name__ == '__main__':
    main()
