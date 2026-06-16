"""
OPM System Configuration
========================

Fuente única de verdad para todos los parámetros de hardware y óptica.
Importar desde acá en lugar de definir constantes en cada módulo.

Modificar acá cuando cambie un puerto, canal DAQ, objetivo, etc.
"""

# =============================================================================
# Hardware — puertos y canales
# =============================================================================
HARDWARE_CONFIG = {
    'laser_488_port':     'COM4',
    'laser_561_port':     'COM3',
    'stage_port':         'COM7',
    'filter_wheel_serial':'26006458',
    'flipper_illumination_serial': '37009524',  # MFF101 — cambia modo iluminación
    'flipper_detection_serial':    '37009525',  # MFF101 — cambia modo detección
    'daq_num_channels':   3,
    'daq_rate':           1e3,
    # Mapeo de canales DAQ: índice físico AO para cada función
    'channel_camera_trigger': 0,
    'channel_galvo':          2,
}

# =============================================================================
# Óptica y adquisición
# =============================================================================
DEFAULT_CONFIG = {
    'sample_px_um':       0.127,   # tamaño de píxel de la cámara [µm]
    'tilt_deg':           41.0,    # ángulo del plano oblicuo [°]
    # Paso del galvo: definido por criterio de Nyquist (≤ resolución_axial / 2).
    # Resolución axial ~300 nm  →  Nyquist: 150 nm.
    # Mínimo cuantizable a 1 px entero = 1 × 0.127 / cos(41°) ≈ 0.168 µm.
    'galvo_step_um':      1,   # paso entre planos oblicuos [µm]
    'width_px':           2060,    # ancho total del sensor [px]
    'height_px':          2048,    # alto total del sensor [px]
    'galvo_volts_per_um': 0.1 / 6.15,  # calibración del galvo [V/µm]
    'drift_threshold':    1.3,     # desplazamiento mínimo para corregir [µm]
    'safety_xy':          10,      # corrección máxima en XY [µm]
    'safety_z':           8,       # corrección máxima en Z [µm]
    # Factores de conversión para la corrección de drift
    # (µm por píxel en el volumen deskewed)
    'drift_z_um_per_px':  0.130,
    'drift_xy_um_per_px': 0.127,
}
