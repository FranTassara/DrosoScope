"""
OPM System Configuration
========================

Single source of truth for all hardware and optical parameters.
Import from here instead of defining constants in individual modules.

Update this file whenever a COM port, DAQ channel, objective, or
optical parameter changes.
"""

# =============================================================================
# Hardware — ports and channels
# =============================================================================
HARDWARE_CONFIG = {
    'laser_488_port':     'COM4',
    'laser_561_port':     'COM3',
    'stage_port':         'COM7',
    'filter_wheel_serial':'26006458',
    'flipper_illumination_serial': '37009524',  # MFF101 — switches illumination mode
    'flipper_detection_serial':    '37009525',  # MFF101 — switches detection mode
    'daq_num_channels':   3,
    'daq_rate':           1e3,
    # DAQ channel mapping: physical AO index for each function
    'channel_camera_trigger': 0,
    'channel_galvo':          2,
}

# =============================================================================
# Optics and acquisition
# =============================================================================
DEFAULT_CONFIG = {
    'sample_px_um':       0.127,   # camera pixel size [µm]
    'tilt_deg':           41.0,    # oblique plane angle [°]
    # Galvo step: set by Nyquist criterion (≤ axial_resolution / 2).
    # Axial resolution ~300 nm  →  Nyquist: 150 nm.
    # Minimum quantizable to 1 whole pixel = 1 × 0.127 / cos(41°) ≈ 0.168 µm.
    'galvo_step_um':      1,       # step between oblique planes [µm]
    'width_px':           2060,    # full sensor width [px]
    'height_px':          2048,    # full sensor height [px]
    'galvo_volts_per_um': 0.1 / 6.15,  # galvo calibration [V/µm]
    'drift_threshold':    1.3,     # minimum displacement to trigger correction [µm]
    'safety_xy':          10,      # maximum XY drift correction [µm]
    'safety_z':           8,       # maximum Z drift correction [µm]
    # Conversion factors for drift correction
    # (µm per pixel in the deskewed volume)
    'drift_z_um_per_px':  0.130,
    'drift_xy_um_per_px': 0.127,
}
