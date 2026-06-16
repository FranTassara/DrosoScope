# DrosoScope

Control and post-processing software for **DrosoScope**, a Single Objective Light Sheet (SOLS) / Oblique Plane Microscopy (OPM) system built for live *Drosophila melanogaster* imaging.

## Hardware

| Component | Model |
|-----------|-------|
| Camera | PCO.edge 4.2 sCMOS |
| Stage | ASI MS-2000 XYZ motorized stage |
| Laser 488 nm | Oxxius LBX 488 |
| Laser 561 nm | Oxxius LCX 561 |
| DAQ | National Instruments PCIe-6363 |
| Filter wheel | Thorlabs FW103 (6 positions) |
| Flip mirrors | Thorlabs MFF101 × 2 |

> The software also supports a **Thorlabs TLCamera** as a secondary camera (e.g. for live-view testing without the PCO).

## Software dependencies

- Python 3.9+
- PyQt5
- numpy, scipy, scikit-image
- tifffile
- numba
- pyqtgraph
- qdarkstyle
- pylablib (Thorlabs TLCamera)
- pythonnet / clr (Thorlabs Kinesis — filter wheel and flippers)
- pyserial (lasers, stage)
- NI-DAQmx drivers (National Instruments)
- PCO SDK (included in `controllers/pco_sdk/`)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Thorlabs Kinesis and NI-DAQmx must be installed separately as Windows drivers.

## Project structure

```
DrosoScope/
├── main.py                        # Main GUI application (PyQt5)
├── acquisition.py                 # Acquisition worker threads
├── config.py                      # Hardware and optical parameters
├── controllers/                   # Hardware device drivers
│   ├── laser_488nm.py             # Oxxius LBX 488 laser
│   ├── laser_561nm.py             # Oxxius LCX 561 laser
│   ├── stage.py                   # ASI MS-2000 stage
│   ├── filter_wheel.py            # Thorlabs FW103 filter wheel
│   ├── flipper.py                 # Thorlabs MFF101 flip mirrors
│   ├── daq.py                     # NI PCIe-6363 DAQ
│   ├── camera_pco.py              # PCO.edge 4.2 camera
│   ├── camera_pco_liveview_test.py# Standalone PCO live-view test GUI
│   └── pco_sdk/                   # PCO SDK DLLs (Windows, 64-bit)
├── postprocessing/                # Image post-processing pipeline
│   ├── deskew.py                  # OPM deskewing (Numba JIT)
│   ├── pipeline.py                # Full post-processing pipeline (CLI + API)
│   ├── flatfield.py               # Flatfield / illumination correction
│   ├── deconvolution.py           # Richardson-Lucy 3D deconvolution
│   └── run_deconvolution.py       # Batch deconvolution script
├── scripts/
│   └── crop_deskew_interactive.py # Interactive crop + deskew GUI (Tkinter)
└── ui/
    └── MainApp.ui                 # Qt Designer UI layout
```

## Quick start

### 1. Configure hardware ports

Edit [`config.py`](config.py) to match your hardware:

```python
HARDWARE_CONFIG = {
    'laser_488_port': 'COM4',   # Oxxius 488 serial port
    'laser_561_port': 'COM3',   # Oxxius 561 serial port
    'stage_port':     'COM7',   # ASI stage serial port
    'filter_wheel_serial': '26006458',      # Thorlabs FW103 serial number
    'flipper_illumination_serial': '37009524',
    'flipper_detection_serial':    '37009525',
    ...
}
```

### 2. Launch the main application

```bash
python main.py
```

### 3. Post-process acquired data

**Full pipeline (flatfield + deconvolution + deskew):**

```bash
python -m postprocessing.pipeline data/data_crop_488_0.tif --scan-range 70 \
    --flatfield --deconvolve --wavelength 488 -o results/
```

**Interactive crop + deskew:**

```bash
python scripts/crop_deskew_interactive.py
```

## Optical parameters

The system parameters are defined in `config.py` and used throughout:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `tilt_deg` | 41.0° | Oblique plane angle |
| `sample_px_um` | 0.127 µm | Camera pixel size at sample |
| `galvo_step_um` | 1 µm | Galvo step between planes |
| `NA` | 1.1 | Objective numerical aperture |

## Deskewing algorithm

The deskewing implementation (`postprocessing/deskew.py`) is based on the algorithm from the [Shepherd Lab / QI2lab](https://github.com/QI2lab/OPM), with the following improvements:

- Z-downsampling by **averaging** (not frame-skipping) for better SNR
- Automatic padding to multiples of 4 for cache alignment
- `uint16` output to preserve full dynamic range

## Acquisition features

- **Multi-channel interleaved acquisition** with independent temporal schedulers per channel
- **Z-stack** with motorized stage
- **Drift correction** via phase cross-correlation between consecutive volumes
- Real-time deskew and max projection during acquisition
- Metadata export with acquisition parameters
