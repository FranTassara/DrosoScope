"""
OPM Autofocus GUI
====================
GUI para autofoco en tiempo real basada en el paper
"Active remote focus stabilization in oblique plane microscopy"
Migrada a Pyside

Hardware:
- Cámara: Thorlabs Zelux
- Piezo: Thorlabs PFM450 (via Kinesis)

Autor: Francisco - Modificaciones por Tomas
Fecha: 2026
"""

import numpy as np
import cv2
import clr
import sys
import time
import threading
import os
import json
import csv
from datetime import datetime
from collections import deque

#from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QGroupBox, QTabWidget,
    QCheckBox, QComboBox, QSplitter, QFrame, QFileDialog, QMessageBox, QDialog,
    QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QFont

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# ============================================================================
# HARDWARE — Kinesis DLLs
# ============================================================================
#sys.path.append(r'C:\Program Files\Thorlabs\Kinesis')
#clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll")
#clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericPiezoCLI.dll")
#clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.Benchtop.PrecisionPiezoCLI.dll")

#from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
#from Thorlabs.MotionControl.Benchtop.PrecisionPiezoCLI import BenchtopPrecisionPiezo
#from Thorlabs.MotionControl.GenericPiezoCLI import Piezo
#from System import Decimal

# ============================================================================
# CONFIGURACIÓN POR DEFECTO
# ============================================================================
DEFAULT_CONFIG = {
    'piezo_serial': "44515714",
    'piezo_channel': 1,
    'camera_serial': "33339",
    'exposure_us': 10000,
    'roi': {'x': 0, 'y': 0, 'width': 0, 'height': 0},
    'acquisition_interval_ms': 500,
    'piezo_step_um': 1.0,
    'slope_angle': 0.0,
    'slope_intensity': 0.0,
    'pid_kp': 0.1,
    'pid_ki': 0.01,
    'pid_kd': 0.0,
    'piezo_min_um': 50.0,
    'piezo_max_um': 420.0,
    'psf_method': 'pca',
    'deadband_nm': 50,
    'control_mode': 'angle',
    'calib_save_format': 'JSON'
}

CONFIG_FILENAME = "autofocus_config.json"

# ============================================================================
# 1. FUNCIONES AUXILIARES  
# ============================================================================

def calculate_robust_moments(image):
    """Calcula el centro de masas, ángulo e intensidad robusta."""
    clean_image = image.astype(np.float64)
    M = np.sum(clean_image)
    if M == 0:
        return 0, 0, 0, 0

    y_dim, x_dim = clean_image.shape
    x, y = np.arange(x_dim), np.arange(y_dim)
    X, Y = np.meshgrid(x, y)

    x_c = np.sum(X * clean_image) / M
    y_c = np.sum(Y * clean_image) / M

    mu_xx = np.sum((X - x_c)**2 * clean_image) / M
    mu_yy = np.sum((Y - y_c)**2 * clean_image) / M
    mu_xy = np.sum((X - x_c) * (Y - y_c) * clean_image) / M
    theta = 0.5 * np.arctan2(2 * mu_xy, mu_xx - mu_yy)

    arr_ord = np.sort(clean_image.flatten())
    percentindex = max(1, int(len(arr_ord) * 0.01))
    mean_int = np.mean(arr_ord[-percentindex:])

    return x_c, y_c, theta, mean_int


def calculate_psf_center(image, roi_size=50):
    """Extrae la ROI y calcula los parámetros del spot."""
    if image is None or image.size == 0:
        return None, None, None, None

    y_bright, x_bright = np.unravel_index(np.argmax(image, axis=None), image.shape)

    background_level = np.median(image)
    image_clean = np.clip(image - background_level, 0, None)

    x_min = max(0, x_bright - roi_size // 2)
    x_max = min(image_clean.shape[1], x_bright + roi_size // 2)
    y_min = max(0, y_bright - roi_size // 2)
    y_max = min(image_clean.shape[0], y_bright + roi_size // 2)

    roi = image_clean[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None, None, None, None

    x_com_roi, y_com_roi, angle, intensity = calculate_robust_moments(roi)
    return x_min + x_com_roi, y_min + y_com_roi, angle, intensity


# ============================================================================
# 2. CONTROLADOR PID  
# ============================================================================

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def update(self, error_z_um, dt):
        p_term = self.kp * error_z_um
        self.integral += error_z_um * dt
        i_term = self.ki * self.integral
        derivative = (error_z_um - self.previous_error) / dt if dt > 0 else 0
        d_term = self.kd * derivative
        control_signal = p_term + i_term + d_term
        self.previous_error = error_z_um
        return control_signal

    def reset(self):
        self.previous_error = 0
        self.integral = 0

    def update_gains(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd


# ============================================================================
# 3. LOGGER  
# ============================================================================

class AutoFocusLogger:
    def __init__(self):
        self.logs = []
        self.start_time = time.time()

    def log(self, cx, cy, angle, intensity, err_ang, err_int, unified_z_error, control_signal_um, piezo_pos):
        elapsed = time.time() - self.start_time
        self.logs.append([
            round(elapsed, 3),
            round(cx, 3) if cx is not None else 0,
            round(cy, 3) if cy is not None else 0,
            round(float(angle), 5) if angle is not None else 0,
            round(float(intensity), 1) if intensity is not None else 0,
            round(float(err_ang), 5) if err_ang is not None else 0,
            round(float(err_int), 1) if err_int is not None else 0,
            round(unified_z_error, 4),
            round(control_signal_um, 4),
            round(piezo_pos, 3)
        ])

    def save(self, filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s', 'cx_px', 'cy_px', 'angle_rad', 'intensity_cts',
                             'err_angle_rad', 'err_intensity_cts', 'error_z_um',
                             'control_signal_um', 'piezo_um'])
            writer.writerows(self.logs)
        return filename

    def clear(self):
        self.logs = []
        self.start_time = time.time()


# ============================================================================
# STYLESHEET
# ============================================================================
STYLESHEET = """
/* Paleta:
   bg principal  #0d0d14   (casi negro azulado)
   superficie    #14141f   (paneles, tabs)
   widget        #1c1c2e   (entries, combos)
   borde         #2e2e45
   borde hover   #4a4a6a
   texto         #dcdcf0
   texto tenue   #6a6a8a
   acento azul   #7aa2f7
   verde         #9ece6a
   rojo          #f7768e
   naranja       #ff9e64
   amarillo      #e0af68
   violeta pos.  #bb9af7   (posición piezo)
*/

QMainWindow, QDialog {
    background-color: #0d0d14;
}
QWidget {
    background-color: #0d0d14;
    color: #dcdcf0;
    font-family: "Segoe UI", "Arial";
    font-size: 10pt;
}
QGroupBox {
    border: 1px solid #2e2e45;
    border-radius: 6px;
    margin-top: 10px;
    padding: 8px 4px 4px 4px;
    font-weight: bold;
    color: #7aa2f7;
    background-color: #14141f;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    background-color: #14141f;
}
QTabWidget::pane {
    border: 1px solid #2e2e45;
    border-radius: 4px;
    background-color: #14141f;
    top: -1px;
}
QTabBar::tab {
    background: #14141f;
    color: #6a6a8a;
    padding: 6px 14px;
    border: 1px solid #2e2e45;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #1c1c2e;
    color: #dcdcf0;
    border-bottom-color: #1c1c2e;
}
QTabBar::tab:hover:!selected {
    background: #1a1a28;
    color: #dcdcf0;
}
QPushButton {
    background-color: #1c1c2e;
    color: #dcdcf0;
    border: 1px solid #2e2e45;
    border-radius: 5px;
    padding: 5px 12px;
    min-height: 22px;
}
QPushButton:hover {
    background-color: #252540;
    border-color: #4a4a6a;
}
QPushButton:pressed {
    background-color: #2e2e50;
}
QPushButton:disabled {
    background-color: #0d0d14;
    color: #3a3a55;
    border-color: #1c1c2e;
}

/* Botón de ayuda (?) */
QPushButton#btn_help {
    background-color: #313244;
    color: #89b4fa;
    border-radius: 12px;
    font-weight: bold;
    padding: 0px;
}
QPushButton#btn_help:hover { background-color: #45475a; border-color: #89b4fa; }

/* Azul — conectar hardware */
QPushButton#btn_connect {
    background-color: #152040;
    color: #7aa2f7;
    border-color: #7aa2f7;
    font-weight: bold;
}
QPushButton#btn_connect:hover { background-color: #1a2b55; }
QPushButton#btn_connect:disabled { background-color: #0d0d14; color: #3a3a55; border-color: #1c1c2e; }

/* Verde — preview / start */
QPushButton#btn_green {
    background-color: #152212;
    color: #9ece6a;
    border-color: #9ece6a;
    font-weight: bold;
}
QPushButton#btn_green:hover { background-color: #1d3018; }
QPushButton#btn_green:disabled { background-color: #0d0d14; color: #3a3a55; border-color: #1c1c2e; }

/* Rojo — stop */
QPushButton#btn_red {
    background-color: #2a1020;
    color: #f7768e;
    border-color: #f7768e;
    font-weight: bold;
}
QPushButton#btn_red:hover { background-color: #38152a; }
QPushButton#btn_red:disabled { background-color: #0d0d14; color: #3a3a55; border-color: #1c1c2e; }

/* Naranja — calibración */
QPushButton#btn_orange {
    background-color: #2a1a0a;
    color: #ff9e64;
    border-color: #ff9e64;
}
QPushButton#btn_orange:hover { background-color: #38230f; }
QPushButton#btn_orange:disabled { background-color: #0d0d14; color: #3a3a55; border-color: #1c1c2e; }

/* Amarillo — setpoint / maxint */
QPushButton#btn_yellow {
    background-color: #242010;
    color: #e0af68;
    border-color: #e0af68;
}
QPushButton#btn_yellow:hover { background-color: #302c18; }
QPushButton#btn_yellow:disabled { background-color: #0d0d14; color: #3a3a55; border-color: #1c1c2e; }

QLineEdit {
    background-color: #1c1c2e;
    color: #dcdcf0;
    border: 1px solid #2e2e45;
    border-radius: 4px;
    padding: 3px 6px;
    selection-background-color: #7aa2f7;
}
QLineEdit:focus { border-color: #7aa2f7; }

QTextEdit {
    background-color: #0a0a10;
    color: #9ece6a;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 9pt;
    border: 1px solid #2e2e45;
    border-radius: 4px;
}
QComboBox {
    background-color: #1c1c2e;
    color: #dcdcf0;
    border: 1px solid #2e2e45;
    border-radius: 4px;
    padding: 3px 6px;
    min-height: 22px;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #1c1c2e;
    color: #dcdcf0;
    selection-background-color: #2e2e50;
    border: 1px solid #2e2e45;
}
QCheckBox { color: #dcdcf0; spacing: 6px; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #2e2e45;
    border-radius: 3px;
    background: #1c1c2e;
}
QCheckBox::indicator:checked { background: #7aa2f7; border-color: #7aa2f7; }

QScrollBar:vertical {
    background: #0d0d14; width: 8px; border-radius: 4px;
}
QScrollBar::handle:vertical { background: #2e2e45; border-radius: 4px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background: #4a4a6a; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QSplitter::handle { background: #2e2e45; width: 2px; }
"""


# ============================================================================
# DIÁLOGO DE CURVA DE CALIBRACIÓN
# ============================================================================

class CalibrationPlotDialog(QDialog):
    """Muestra temporalmente la curva de calibración tras la calibración automática."""

    def __init__(self, calib_data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Curva de Calibración")
        self.setMinimumSize(700, 500)
        self.setAttribute(Qt.WA_DeleteOnClose)

        z   = calib_data['z_um']
        ang = calib_data['angle']
        ity = calib_data['intensity']
        slope_ang = calib_data['slope_angle']
        slope_ity = calib_data['slope_intensity']

        # Rectas ajustadas
        z_fit  = np.linspace(z.min(), z.max(), 200)
        z_mid  = z.mean()
        ang_fit = slope_ang * (z_fit - z_mid) + ang.mean()
        ity_fit = slope_ity * (z_fit - z_mid) + ity.mean()

        fig = Figure(figsize=(7, 4.5), tight_layout=True)
        fig.patch.set_facecolor('#1a1b26')

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        for ax in (ax1, ax2):
            ax.set_facecolor('#1a1b26')
            ax.tick_params(colors='#c0caf5')
            ax.xaxis.label.set_color('#c0caf5')
            ax.yaxis.label.set_color('#c0caf5')
            ax.title.set_color('#c0caf5')
            for spine in ax.spines.values():
                spine.set_edgecolor('#3b3d5c')

        ax1.scatter(z, np.degrees(ang), s=18, color='#7aa2f7', zorder=3, label='datos')
        ax1.plot(z_fit, np.degrees(ang_fit), color='#ff9e64', lw=1.5,
                 label=f'ajuste  {np.degrees(slope_ang)*1e3:.2f} m°/µm')
        ax1.set_xlabel('Piezo (µm)')
        ax1.set_ylabel('Ángulo (°)')
        ax1.set_title('Ángulo vs Posición')
        ax1.legend(fontsize=7, facecolor='#252535', labelcolor='#c0caf5', edgecolor='#3b3d5c')

        ax2.scatter(z, ity, s=18, color='#9ece6a', zorder=3, label='datos')
        ax2.plot(z_fit, ity_fit, color='#ff9e64', lw=1.5,
                 label=f'ajuste  {slope_ity:.1f} cts/µm')
        ax2.set_xlabel('Piezo (µm)')
        ax2.set_ylabel('Intensidad (cts)')
        ax2.set_title('Intensidad vs Posición')
        ax2.legend(fontsize=7, facecolor='#252535', labelcolor='#c0caf5', edgecolor='#3b3d5c')

        canvas = FigureCanvas(fig)

        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(canvas)
        layout.addWidget(btn_close)


# ============================================================================
# DIÁLOGO DE AYUDA
# ============================================================================

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Guía de Operación")
        self.setMinimumSize(600, 550)
        self.setAttribute(Qt.WA_DeleteOnClose)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet("background-color: #14141f;")

        help_label = QLabel()
        help_label.setTextFormat(Qt.RichText)
        help_label.setWordWrap(True)
        help_label.setOpenExternalLinks(True)
        help_label.setStyleSheet("""
            QLabel {
                background-color: #14141f;
                padding: 15px;
                font-size: 10pt;
            }
            h3 {
                color: #7aa2f7; /* acento azul */
                font-size: 13pt;
                font-weight: bold;
                margin-top: 10px;
                margin-bottom: 5px;
                border-bottom: 1px solid #2e2e45;
            }
            h4 {
                color: #9ece6a; /* verde */
                font-size: 11pt;
                font-weight: bold;
                margin-top: 8px;
                margin-bottom: 3px;
            }
            p, ol, ul { margin-bottom: 8px; }
            ul, ol { margin-left: 20px; }
            li { margin-bottom: 4px; }
            code {
                background-color: #1c1c2e;
                color: #ff9e64; /* naranja */
                padding: 2px 4px;
                border-radius: 3px;
                font-family: "Consolas", "Courier New", monospace;
            }
        """)

        help_text = """
        <h3>Flujo de Interacción del Sistema</h3>

        <h4>1. Configuración Inicial (Setup)</h4>
        <ol>
            <li>Conectar el hardware: Camara y Piezo. Decidir un ROI de trabajo adecuado para la cámara (idealmente centrado en el spot láser).</li>
            <li>Ingresar las ganancias PID deseadas en el grupo <b>Parámetros de Control</b>.</li>
            <li>En la pestaña de <i>Calibración</i>, presionar el botón <b>Ejecutar Calibración</b> para establecer la respuesta del sistema y la sensibilidad inicial del spot láser.</li>
            <li>En la pestaña de <i>Autofoco</i>, presionar <b>Establecer Setpoint</b> para definir la posición actual del spot como la referencia de enfoque (punto cero).</li>
        </ol>

        <h4>2. Operación</h4>
        <ol>
            <li>La transmisión en tiempo real en el grupo de <b>Vista de Cámara</b> proporciona retroalimentación visual constante de la PSF (Point Spread Function).</li>
            <li>Presionar el botón <b>INICIAR</b> para activar el lazo de control. El sistema corregirá automáticamente las derivas térmicas o mecánicas respecto al setpoint.</li>
        </ol>

        <h4>3. Ajustes y Recalibración</h4>
        <ol>
            <li><b>Actualización de parámetros:</b> Presionar el botón <code>DETENER</code> -> Ajustar las ganancias PID -> Presionar el botón <code>INICIAR</code>.</li>
            <li><b>Recalibración del sistema:</b> En caso de ser necesario, presionar <code>DETENER</code> -> Presionar el botón <code>RESET</code> y repetir el proceso de <b>Configuración Inicial</b> para establecer una nueva posición de referencia.</li>
        </ol>

        <h3>Protocolo de Guardado de Datos</h3>
        <ol>
            <li><b>Calibración (JSON/CSV):</b> Se genera automáticamente al finalizar un barrido. El formato <code>JSON</code> es mandatorio si se desea recargar la curva de respuesta en sesiones futuras.</li>
            <li><b>Log CSV:</b> Registra el historial completo de la sesión de autofoco (ideal para análisis de estabilidad a largo plazo).</li>
            <li><b>Datos de Error:</b> Guarda únicamente los últimos 300 puntos visualizados en los gráficos de la interfaz.</li>
        </ol>
        """
        help_label.setText(help_text)

        scroll_area.setWidget(help_label)
        main_layout.addWidget(scroll_area)

        close_button = QPushButton("Cerrar")
        close_button.clicked.connect(self.accept)
        main_layout.addWidget(close_button, 0, Qt.AlignRight)


# ============================================================================
# GUI PRINCIPAL
# ============================================================================

class AutofocusGUI(QMainWindow):

    # Señales para comunicación hilo-de-fondo → hilo-UI
    sig_log = Signal(str)
    sig_saturation = Signal()
    sig_calib_finished = Signal(bool, str)       # (éxito, mensaje_final)
    sig_calib_data = Signal(object)              # dict con arrays de calibración para graficar
    sig_maxint_finished = Signal(bool, float, float, str)  # (éxito, pos, val, mensaje)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OPM Autofoco")
        self.resize(1400, 900)

        # --- Configuración ---
        self.config = self.load_config()

        # --- Hardware ---
        self.piezo_device = None
        self.piezo_channel = None
        self.sdk = None
        self.camera = None
        self.pid = None
        self.logger = AutoFocusLogger()

        # --- Estado de conexión ---
        self.piezo_connected = False
        self.camera_connected = False

        # --- Estado del sistema ---
        self.running = False
        self.calibrating = False
        self.calibrated = False
        self.preview_active = False
        self.saturation_threshold = 1023
        self.saturation_counter = 0

        # --- Datos del spot ---
        self.setpoint_x = None
        self.setpoint_y = None
        self.setpoint_angle = None
        self.setpoint_intensity = None
        self.current_image = None
        self.current_cx = None
        self.current_cy = None
        self.current_angle = None
        self.current_intensity = None
        self.current_piezo_pos = 0

        # --- Parámetros y sensibilidades ---
        self.slope_angle = self.config.get('slope_angle', 0.0)
        self.slope_intensity = self.config.get('slope_intensity', 0.0)
        self.acquisition_interval_ms = self.config.get('acquisition_interval_ms', 500)
        self.psf_method = self.config.get('psf_method', 'pca')
        self.camera_roi = self.config.get('roi', {'x': 0, 'y': 0, 'width': 0, 'height': 0})
        self.use_angle = self.config.get('use_angle', False)
        self.use_intensity = self.config.get('use_intensity', False)
        self.calib_slope_angle = 0.0
        self.calib_slope_intensity = 0.0
        self.diagonal_direction = np.array([1.0, 0.0, 0.0, 0.0])
        self.sensitivity = self.config.get('sensitivity', 1.0)
        self.control_mode = self.config.get('control_mode', 'angle')

        # --- Thread de adquisición ---
        self.acquisition_running = False
        self.image_lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self._acq_thread = None

        # Valor cacheado del intervalo: los threads de fondo lo leen aquí
        # (nunca leen directamente el widget desde otro hilo)
        self._acq_interval_s = self.acquisition_interval_ms / 1000.0

        # --- Datos para gráficos ---
        self.max_points = 300
        self.time_data = deque(maxlen=self.max_points)
        self.piezo_data = deque(maxlen=self.max_points)
        self.angle_data = deque(maxlen=self.max_points)
        self.intensity_data = deque(maxlen=self.max_points)
        self.err_ang_data = deque(maxlen=self.max_points)
        self.err_int_data = deque(maxlen=self.max_points)
        self.z_err_unified_data = deque(maxlen=self.max_points)
        self.z_err_ang_data = deque(maxlen=self.max_points)
        self.z_err_int_data = deque(maxlen=self.max_points)
        self.start_time = time.time()

        # --- FPS ---
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0

        # --- Límites del piezo ---
        self.pos_min_um = self.config.get('piezo_min_um', 50.0)
        self.pos_max_um = self.config.get('piezo_max_um', 300.0)

        # --- Construir UI ---
        self.create_widgets()

        # --- Conectar señales ---
        self.sig_log.connect(self.log_status)
        self.sig_saturation.connect(self._handle_saturation)
        self.sig_calib_finished.connect(self._on_calib_finished)
        self.sig_calib_data.connect(self._show_calib_plot)
        self.sig_maxint_finished.connect(self._on_maxint_finished)

        # --- Timers ---
        self.piezo_timer = QTimer(self)
        self.piezo_timer.timeout.connect(self.update_piezo_display)
        self.piezo_timer.start(250)

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.update_preview)

        self.plots_timer = QTimer(self)
        self.plots_timer.timeout.connect(self.update_plots)

    # =========================================================================
    # CONFIGURACIÓN
    # =========================================================================

    def load_config(self):
        config = DEFAULT_CONFIG.copy()
        if os.path.exists(CONFIG_FILENAME):
            try:
                with open(CONFIG_FILENAME, 'r') as f:
                    config.update(json.load(f))
            except Exception:
                pass
        return config

    def save_config(self):
        try:
            config = {
                'piezo_serial': self.entry_piezo_serial.text(),
                'piezo_channel': self.config.get('piezo_channel', DEFAULT_CONFIG['piezo_channel']),
                'camera_serial': self.entry_camera_serial.text(),
                'exposure_us': int(self.entry_exposure.text()),
                'roi': self.camera_roi,
                'acquisition_interval_ms': int(self.entry_acq_interval.text()),
                'piezo_step_um': float(self.entry_step.text()),
                'slope_angle': self.slope_angle,
                'slope_intensity': self.slope_intensity,
                'pid_kp': float(self.entry_kp.text()),
                'pid_ki': float(self.entry_ki.text()),
                'pid_kd': float(self.entry_kd.text()),
                'piezo_min_um': float(self.entry_piezo_min.text()),
                'piezo_max_um': float(self.entry_piezo_max.text()),
                'psf_method': self.psf_method,
                'deadband_nm': float(self.entry_deadband.text()),
                'control_mode': self.control_mode,
                'calib_save_format': self.combo_calib_format.currentText()
            }
            with open(CONFIG_FILENAME, 'w') as f:
                json.dump(config, f, indent=4)
            self.log_status(f"✓ Configuración guardada en {CONFIG_FILENAME}")
        except Exception as e:
            self.log_status(f"✗ Error guardando config: {e}")

    # =========================================================================
    # CONSTRUCCIÓN DE WIDGETS
    # =========================================================================

    def create_widgets(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        left_widget = QWidget()
        left_widget.setMinimumWidth(340)
        left_widget.setMaximumWidth(460)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([380, 1020])

        self._build_left_panel(left_layout)
        self._build_right_panel(right_layout)

    def _build_left_panel(self, layout):
        header_layout = QHBoxLayout()
        header_layout.addSpacing(24)  # Espaciador invisible para compensar el botón y centrar el título
        header_layout.addStretch()
        
        title = QLabel("Controles")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        self.btn_global_help = QPushButton("?")
        self.btn_global_help.setObjectName("btn_help")
        self.btn_global_help.setFixedSize(24, 24)
        self.btn_global_help.clicked.connect(self.show_global_save_help)
        header_layout.addWidget(self.btn_global_help)

        layout.addLayout(header_layout)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, stretch=1)

        self._build_hardware_tab()
        self._build_calib_tab()
        self._build_autofocus_tab()
        self._build_config_tab()

        # --- Log ---
        log_label = QLabel("Log")
        log_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        layout.addWidget(log_label)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setFixedHeight(160)
        layout.addWidget(self.status_text)

    def _build_hardware_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(8)

        # ── Piezo ──────────────────────────────────────────────────────────────
        piezo_group = QGroupBox("Piezo")
        pg = QVBoxLayout(piezo_group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Serial:"))
        self.entry_piezo_serial = QLineEdit(self.config.get('piezo_serial', DEFAULT_CONFIG['piezo_serial']))
        self.entry_piezo_serial.setFixedWidth(110)
        row.addWidget(self.entry_piezo_serial)
        row.addStretch()
        pg.addLayout(row)

        self.btn_connect_piezo = QPushButton("Conectar Piezo")
        self.btn_connect_piezo.setObjectName("btn_connect")
        self.btn_connect_piezo.clicked.connect(self.connect_piezo)
        pg.addWidget(self.btn_connect_piezo)

        self.piezo_status = QLabel("● Desconectado")
        self.piezo_status.setStyleSheet("color: #f38ba8;")
        self.piezo_status.setAlignment(Qt.AlignCenter)
        pg.addWidget(self.piezo_status)

        self.piezo_pos_label = QLabel("Posición: -- µm")
        self.piezo_pos_label.setFont(QFont("Consolas", 11, QFont.Bold))
        self.piezo_pos_label.setAlignment(Qt.AlignCenter)
        self.piezo_pos_label.setStyleSheet("color: #cba6f7;")
        pg.addWidget(self.piezo_pos_label)

        manual_row = QHBoxLayout()
        self.btn_piezo_down = QPushButton("◄")
        self.btn_piezo_down.setFixedWidth(36)
        self.btn_piezo_down.clicked.connect(lambda: self.move_piezo(-1))
        self.btn_piezo_down.setEnabled(False)
        manual_row.addWidget(self.btn_piezo_down)
        manual_row.addWidget(QLabel("Step (µm):"))
        self.entry_step = QLineEdit(str(self.config.get('piezo_step_um', 1.0)))
        self.entry_step.setFixedWidth(60)
        manual_row.addWidget(self.entry_step)
        self.btn_piezo_up = QPushButton("►")
        self.btn_piezo_up.setFixedWidth(36)
        self.btn_piezo_up.clicked.connect(lambda: self.move_piezo(1))
        self.btn_piezo_up.setEnabled(False)
        manual_row.addWidget(self.btn_piezo_up)
        manual_row.addStretch()
        pg.addLayout(manual_row)

        home_pos = (self.pos_min_um + self.pos_max_um) / 2
        self.btn_home = QPushButton(f"Home ({home_pos:.0f} µm)")
        self.btn_home.clicked.connect(self.piezo_home)
        self.btn_home.setEnabled(False)
        pg.addWidget(self.btn_home)

        goto_row = QHBoxLayout()
        goto_row.addWidget(QLabel("Ir a:"))
        self.entry_goto = QLineEdit("225")
        self.entry_goto.setFixedWidth(70)
        goto_row.addWidget(self.entry_goto)
        goto_row.addWidget(QLabel("µm"))
        self.btn_goto = QPushButton("Ir")
        self.btn_goto.setFixedWidth(40)
        self.btn_goto.clicked.connect(self.piezo_goto)
        self.btn_goto.setEnabled(False)
        goto_row.addWidget(self.btn_goto)
        goto_row.addStretch()
        pg.addLayout(goto_row)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #45475a; max-height: 1px;")
        pg.addWidget(sep)

        lim_lbl = QLabel("Límites de Seguridad")
        lim_lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        lim_lbl.setAlignment(Qt.AlignCenter)
        pg.addWidget(lim_lbl)

        lim_grid = QGridLayout()
        lim_grid.addWidget(QLabel("Mín:"), 0, 0, Qt.AlignRight)
        self.entry_piezo_min = QLineEdit(str(self.pos_min_um))
        self.entry_piezo_min.setFixedWidth(65)
        lim_grid.addWidget(self.entry_piezo_min, 0, 1)
        lim_grid.addWidget(QLabel("µm"), 0, 2)
        lim_grid.addWidget(QLabel("Máx:"), 1, 0, Qt.AlignRight)
        self.entry_piezo_max = QLineEdit(str(self.pos_max_um))
        self.entry_piezo_max.setFixedWidth(65)
        lim_grid.addWidget(self.entry_piezo_max, 1, 1)
        lim_grid.addWidget(QLabel("µm"), 1, 2)
        pg.addLayout(lim_grid)

        self.btn_apply_limits = QPushButton("Aplicar Límites")
        self.btn_apply_limits.clicked.connect(self.apply_piezo_limits)
        pg.addWidget(self.btn_apply_limits)

        self.limits_status_label = QLabel(f"Rango: {self.pos_min_um:.0f} - {self.pos_max_um:.0f} µm")
        self.limits_status_label.setStyleSheet("color: #89b4fa; font-size: 8pt;")
        self.limits_status_label.setAlignment(Qt.AlignCenter)
        pg.addWidget(self.limits_status_label)

        layout.addWidget(piezo_group)

        # ── Cámara ─────────────────────────────────────────────────────────────
        camera_group = QGroupBox("Cámara")
        cg = QVBoxLayout(camera_group)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Serial:"))
        self.entry_camera_serial = QLineEdit(self.config.get('camera_serial', DEFAULT_CONFIG['camera_serial']))
        self.entry_camera_serial.setFixedWidth(110)
        cam_row.addWidget(self.entry_camera_serial)
        cam_row.addStretch()
        cg.addLayout(cam_row)

        self.btn_connect_camera = QPushButton("Conectar Cámara")
        self.btn_connect_camera.setObjectName("btn_connect")
        self.btn_connect_camera.clicked.connect(self.connect_camera)
        cg.addWidget(self.btn_connect_camera)

        self.camera_status = QLabel("● Desconectada")
        self.camera_status.setStyleSheet("color: #f38ba8;")
        self.camera_status.setAlignment(Qt.AlignCenter)
        cg.addWidget(self.camera_status)

        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("Exposición (µs):"))
        self.entry_exposure = QLineEdit(str(self.config.get('exposure_us', 10000)))
        self.entry_exposure.setFixedWidth(85)
        exp_row.addWidget(self.entry_exposure)
        self.btn_set_exposure = QPushButton("Aplicar")
        self.btn_set_exposure.clicked.connect(self.set_exposure)
        self.btn_set_exposure.setEnabled(False)
        exp_row.addWidget(self.btn_set_exposure)
        exp_row.addStretch()
        cg.addLayout(exp_row)

        roi_row = QHBoxLayout()
        roi_row.addWidget(QLabel("ROI:"))
        self.btn_set_roi = QPushButton("Configurar")
        self.btn_set_roi.clicked.connect(self.configure_roi)
        self.btn_set_roi.setEnabled(False)
        roi_row.addWidget(self.btn_set_roi)
        self.btn_reset_roi = QPushButton("Reset")
        self.btn_reset_roi.clicked.connect(self.reset_roi)
        self.btn_reset_roi.setEnabled(False)
        roi_row.addWidget(self.btn_reset_roi)
        roi_row.addStretch()
        cg.addLayout(roi_row)

        self.roi_label = QLabel("ROI: Full frame")
        self.roi_label.setStyleSheet("color: #6c7086; font-size: 9pt;")
        self.roi_label.setAlignment(Qt.AlignCenter)
        cg.addWidget(self.roi_label)

        acq_row = QHBoxLayout()
        acq_row.addWidget(QLabel("Intervalo (ms):"))
        self.entry_acq_interval = QLineEdit(str(self.config.get('acquisition_interval_ms', 500)))
        self.entry_acq_interval.setFixedWidth(70)
        self.entry_acq_interval.textChanged.connect(self._update_acq_interval_cache)
        acq_row.addWidget(self.entry_acq_interval)
        acq_row.addWidget(QLabel("(500 ms = 2 Hz)"))
        acq_row.addStretch()
        cg.addLayout(acq_row)

        self.btn_preview = QPushButton("▶  Iniciar Preview")
        self.btn_preview.setObjectName("btn_green")
        self.btn_preview.clicked.connect(self.toggle_preview)
        self.btn_preview.setEnabled(False)
        cg.addWidget(self.btn_preview)

        layout.addWidget(camera_group)
        layout.addStretch()

        scroll.setWidget(container)
        self.tabs.addTab(scroll, "Hardware")

    def _build_calib_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(8)

        # Cargar calibración
        self.btn_load_calib = QPushButton("Cargar Calibración (JSON)")
        self.btn_load_calib.clicked.connect(self.load_calibration)
        layout.addWidget(self.btn_load_calib)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #45475a; max-height: 1px;")
        layout.addWidget(sep)

        # ── Calibración automática ─────────────────────────────────────────────
        auto_group = QGroupBox("Calibración Automática")
        ag = QGridLayout(auto_group)

        ag.addWidget(QLabel("Rango (µm):"), 0, 0, Qt.AlignRight)
        self.entry_calib_range = QLineEdit("5")
        self.entry_calib_range.setFixedWidth(70)
        ag.addWidget(self.entry_calib_range, 0, 1)

        ag.addWidget(QLabel("Pasos:"), 1, 0, Qt.AlignRight)
        self.entry_calib_steps = QLineEdit("21")
        self.entry_calib_steps.setFixedWidth(70)
        ag.addWidget(self.entry_calib_steps, 1, 1)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Guardar como:"))
        self.combo_calib_format = QComboBox()
        self.combo_calib_format.addItems(['CSV', 'JSON'])
        idx = self.combo_calib_format.findText(self.config.get('calib_save_format', 'JSON'))
        if idx >= 0:
            self.combo_calib_format.setCurrentIndex(idx)
        self.combo_calib_format.setFixedWidth(80)
        fmt_row.addWidget(self.combo_calib_format)
        fmt_row.addStretch()
        ag.addLayout(fmt_row, 2, 0, 1, 2)

        self.btn_run_calib = QPushButton("Ejecutar Calibración")
        self.btn_run_calib.setObjectName("btn_orange")
        self.btn_run_calib.clicked.connect(self.run_calibration)
        self.btn_run_calib.setEnabled(False)
        ag.addWidget(self.btn_run_calib, 3, 0, 1, 2)

        layout.addWidget(auto_group)

               # ── Fuente de error PID ────────────────────────────────────────────────
        pid_info = QGroupBox("Fuente de Error (PID)")
        pi = QVBoxLayout(pid_info)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Fuente:"))
        self.combo_control_mode = QComboBox()
        self.combo_control_mode.addItem('Ángulo', 'angle')
        self.combo_control_mode.addItem('Intensidad', 'intensity')
        idx = self.combo_control_mode.findData(self.control_mode)
        if idx >= 0:
            self.combo_control_mode.setCurrentIndex(idx)
        self.combo_control_mode.currentIndexChanged.connect(self._update_control_mode)
        mode_row.addWidget(self.combo_control_mode)
        mode_row.addStretch()
        pi.addLayout(mode_row)

        pi.addWidget(QLabel("(ambos errores se calculan y guardan igualmente)"))

        warn_lbl = QLabel("[!] Requiere calibración previa (auto o manual)")
        warn_lbl.setStyleSheet("color: #6c7086; font-size: 8pt;")
        pi.addWidget(warn_lbl)

        layout.addWidget(pid_info)
        layout.addStretch()

        # ── Búsqueda de máximo de intensidad ───────────────────────────────────
        maxint_group = QGroupBox("Búsqueda de Máximo de Intensidad")
        mg = QGridLayout(maxint_group)

        mg.addWidget(QLabel("Rango (µm):"), 0, 0, Qt.AlignRight)
        self.entry_maxint_range = QLineEdit("5")
        self.entry_maxint_range.setFixedWidth(70)
        mg.addWidget(self.entry_maxint_range, 0, 1)

        mg.addWidget(QLabel("Pasos:"), 1, 0, Qt.AlignRight)
        self.entry_maxint_steps = QLineEdit("21")
        self.entry_maxint_steps.setFixedWidth(70)
        mg.addWidget(self.entry_maxint_steps, 1, 1)

        self.btn_run_maxint = QPushButton("Buscar Máximo")
        self.btn_run_maxint.setObjectName("btn_yellow")
        self.btn_run_maxint.clicked.connect(self.run_max_intensity_search)
        self.btn_run_maxint.setEnabled(False)
        mg.addWidget(self.btn_run_maxint, 2, 0, 1, 2)

        self.lbl_maxint_result = QLabel("Máximo en: --")
        self.lbl_maxint_result.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        self.lbl_maxint_result.setAlignment(Qt.AlignCenter)
        mg.addWidget(self.lbl_maxint_result, 3, 0, 1, 2)

        layout.addWidget(maxint_group)

        # ── Ingreso manual de sensibilidad ─────────────────────────────────────
        manual_group = QGroupBox("Ingreso Manual de Sensibilidad")
        mng = QGridLayout(manual_group)

        mng.addWidget(QLabel("Ángulo (rad/µm):"), 0, 0, Qt.AlignRight)
        self.entry_sens_angle = QLineEdit()
        self.entry_sens_angle.setFixedWidth(110)
        mng.addWidget(self.entry_sens_angle, 0, 1)

        mng.addWidget(QLabel("Intens. (cts/µm):"), 1, 0, Qt.AlignRight)
        self.entry_sens_int = QLineEdit()
        self.entry_sens_int.setFixedWidth(110)
        mng.addWidget(self.entry_sens_int, 1, 1)

        self.btn_set_manual_sens = QPushButton("Aplicar Valores Manuales")
        self.btn_set_manual_sens.clicked.connect(self.set_manual_sensitivity)
        mng.addWidget(self.btn_set_manual_sens, 2, 0, 1, 2)

        layout.addWidget(manual_group)

        # ── Sensibilidades actuales ─────────────────────────────────────────────
        sens_group = QGroupBox("Sensibilidades Actuales")
        sg = QVBoxLayout(sens_group)

        self.lbl_sens_angle = QLabel("Ángulo: -- rad/µm")
        self.lbl_sens_angle.setFont(QFont("Segoe UI", 10, QFont.Bold))
        sg.addWidget(self.lbl_sens_angle)

        self.lbl_sens_int = QLabel("Intensidad: -- cts/µm")
        self.lbl_sens_int.setFont(QFont("Segoe UI", 10, QFont.Bold))
        sg.addWidget(self.lbl_sens_int)

        layout.addWidget(sens_group)

        scroll.setWidget(container)
        self.tabs.addTab(scroll, "Calibración")

    def show_global_save_help(self):
        dialog = HelpDialog(self)
        dialog.exec()

    def _update_control_mode(self, index):
        self.control_mode = self.combo_control_mode.itemData(index)
        self.log_status(f"Fuente de error PID cambiada a: {self.control_mode}")

    def _build_autofocus_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(8)

        # ── Setpoint ───────────────────────────────────────────────────────────
        sp_group = QGroupBox("Setpoint")
        spg = QVBoxLayout(sp_group)

        self.btn_set_setpoint = QPushButton("Establecer Setpoint")
        self.btn_set_setpoint.setObjectName("btn_yellow")
        self.btn_set_setpoint.clicked.connect(self.set_setpoint)
        self.btn_set_setpoint.setEnabled(False)
        spg.addWidget(self.btn_set_setpoint)

        self.setpoint_label = QLabel("Setpoint: --")
        self.setpoint_label.setFont(QFont("Consolas", 10))
        self.setpoint_label.setAlignment(Qt.AlignCenter)
        spg.addWidget(self.setpoint_label)

        self.setpoint_angle_label = QLabel("Ángulo: -- | Intensidad: --")
        self.setpoint_angle_label.setFont(QFont("Consolas", 9))
        self.setpoint_angle_label.setStyleSheet("color: #6c7086;")
        self.setpoint_angle_label.setAlignment(Qt.AlignCenter)
        spg.addWidget(self.setpoint_angle_label)

        layout.addWidget(sp_group)

        # ── Parámetros PID ─────────────────────────────────────────────────────
        pid_group = QGroupBox("Parámetros PID")
        pg = QGridLayout(pid_group)

        pg.addWidget(QLabel("Kp:"), 0, 0, Qt.AlignRight)
        self.entry_kp = QLineEdit(str(self.config.get('pid_kp', 0.1)))
        self.entry_kp.setFixedWidth(80)
        pg.addWidget(self.entry_kp, 0, 1)

        pg.addWidget(QLabel("Ki:"), 1, 0, Qt.AlignRight)
        self.entry_ki = QLineEdit(str(self.config.get('pid_ki', 0.01)))
        self.entry_ki.setFixedWidth(80)
        pg.addWidget(self.entry_ki, 1, 1)

        pg.addWidget(QLabel("Kd:"), 2, 0, Qt.AlignRight)
        self.entry_kd = QLineEdit(str(self.config.get('pid_kd', 0.0)))
        self.entry_kd.setFixedWidth(80)
        pg.addWidget(self.entry_kd, 2, 1)

        layout.addWidget(pid_group)

        # ── Umbral de corrección ───────────────────────────────────────────────
        db_group = QGroupBox("Umbral de Corrección")
        dbg = QVBoxLayout(db_group)

        db_row = QHBoxLayout()
        db_row.addWidget(QLabel("Mov. mínimo (nm):"))
        self.entry_deadband = QLineEdit(str(self.config.get('deadband_nm', 50)))
        self.entry_deadband.setFixedWidth(80)
        db_row.addWidget(self.entry_deadband)
        db_row.addStretch()
        dbg.addLayout(db_row)

        db_hint = QLabel("Solo mueve piezo si |delta| > umbral")
        db_hint.setStyleSheet("color: #6c7086; font-size: 8pt;")
        dbg.addWidget(db_hint)

        layout.addWidget(db_group)

        # ── Solo monitoreo ─────────────────────────────────────────────────────
        self.chk_monitor_only = QCheckBox("Solo monitoreo (no mover piezo)")
        layout.addWidget(self.chk_monitor_only)

        # ── Botones Start / Stop ────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶  INICIAR")
        self.btn_start.setObjectName("btn_green")
        self.btn_start.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.btn_start.setMinimumHeight(40)
        self.btn_start.clicked.connect(self.start_autofocus)
        self.btn_start.setEnabled(False)
        btn_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("■  DETENER")
        self.btn_stop.setObjectName("btn_red")
        self.btn_stop.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.clicked.connect(self.stop_autofocus)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_stop)
        layout.addLayout(btn_row)

        # ── Guardar ────────────────────────────────────────────────────────────
        save_row = QHBoxLayout()
        self.btn_save_log = QPushButton("Guardar Log CSV")
        self.btn_save_log.clicked.connect(self.save_log)
        save_row.addWidget(self.btn_save_log)

        self.btn_save_error_data = QPushButton("Guardar Error Data")
        self.btn_save_error_data.setObjectName("btn_yellow")
        self.btn_save_error_data.clicked.connect(self.save_error_data)
        save_row.addWidget(self.btn_save_error_data)
        layout.addLayout(save_row)

        layout.addStretch()
        scroll.setWidget(container)
        self.tabs.addTab(scroll, "Autofoco")

    def _build_config_tab(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        self.btn_save_config = QPushButton("Guardar Configuración")
        self.btn_save_config.setObjectName("btn_connect")
        self.btn_save_config.clicked.connect(self.save_config)
        layout.addWidget(self.btn_save_config)

        self.btn_load_config = QPushButton("Recargar Configuración")
        self.btn_load_config.clicked.connect(self.reload_config)
        layout.addWidget(self.btn_load_config)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #45475a; max-height: 1px;")
        layout.addWidget(sep)

        info = QLabel(
            "Atajos de teclado:\n"
            "  ← / →  :  Mover piezo\n"
            "  Space   :  Establecer setpoint\n"
            "  S        :  Iniciar / Detener autofoco\n\n"
            "Recomendaciones:\n"
            "  - Intervalo 500 ms  →  2 Hz  (drift lento)\n"
            "  - Intervalo 100 ms  →  10 Hz (drift rápido)\n"
            "  - ROI pequeño  →  más velocidad"
        )
        info.setFont(QFont("Consolas", 9))
        info.setStyleSheet("color: #6c7086;")
        layout.addWidget(info)

        layout.addStretch()
        self.tabs.addTab(container, "Config")

    def _build_right_panel(self, layout):
        title = QLabel("Monitoreo")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # ── Vista de cámara ────────────────────────────────────────────────────
        self.image_label = QLabel()
        self.image_label.setFixedSize(500, 375)
        self.image_label.setStyleSheet("background-color: #000000; border: 1px solid #45475a; border-radius: 4px;")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label, alignment=Qt.AlignHCenter)

        info_row = QHBoxLayout()
        self.spot_info_label = QLabel("Spot: --")
        self.spot_info_label.setFont(QFont("Consolas", 9))
        info_row.addWidget(self.spot_info_label)
        info_row.addStretch()
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setFont(QFont("Consolas", 9))
        info_row.addWidget(self.fps_label)
        layout.addLayout(info_row)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #45475a; max-height: 1px;")
        layout.addWidget(sep)

        # ── Matplotlib ─────────────────────────────────────────────────────────
        self.fig = Figure(figsize=(6, 7), dpi=100)
        self.fig.patch.set_facecolor('#1e1e2e')
        self.fig.subplots_adjust(hspace=0.45)

        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)

        for ax, title_txt, ylabel_txt in [
            (self.ax1, "Posición del Piezo", "Posición (µm)"),
            (self.ax2, "Error del Sistema en Z", "Error (µm)"),
            (self.ax3, "Fuente de Error PID", ""),
        ]:
            ax.set_facecolor('#181825')
            ax.set_title(title_txt, color='#cdd6f4', fontsize=8, pad=4)
            ax.set_ylabel(ylabel_txt, color='#6c7086', fontsize=7)
            ax.tick_params(colors='#6c7086', labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('#45475a')
            ax.grid(True, alpha=0.2, color='#45475a')

        self.ax3.set_xlabel("Tiempo (s)", color='#6c7086', fontsize=7)

        self.line1, = self.ax1.plot([], [], color='#89b4fa', linewidth=1.5)
        self.line2, = self.ax2.plot([], [], color='#f38ba8', linewidth=1.5)
        self.line_sp    = self.ax2.axhline(y=0, color='#cdd6f4', linestyle='-',  linewidth=0.8, alpha=0.7)
        self.line_db_up = self.ax2.axhline(y=0.05, color='#6c7086', linestyle='--', linewidth=0.8, alpha=0.7)
        self.line_db_dn = self.ax2.axhline(y=-0.05, color='#6c7086', linestyle='--', linewidth=0.8, alpha=0.7)
        self.line3a, = self.ax3.plot([], [], color='#a6e3a1', linewidth=1.5)
        self.line3b, = self.ax3.plot([], [], color='#cba6f7', linewidth=1.5)
        self.ax3.legend(loc='upper right', fontsize=7, facecolor='#313244', labelcolor='#cdd6f4')

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #1e1e2e;")
        layout.addWidget(self.canvas, stretch=1)

    # =========================================================================
    # TECLADO
    # =========================================================================

    def keyPressEvent(self, event):
        focused = self.focusWidget()
        if isinstance(focused, QLineEdit):
            super().keyPressEvent(event)
            return
        k = event.key()
        if k == Qt.Key_Left:
            self.move_piezo(-1)
        elif k == Qt.Key_Right:
            self.move_piezo(1)
        elif k == Qt.Key_Space:
            self.set_setpoint()
        elif k in (Qt.Key_S,):
            self.toggle_autofocus()
        else:
            super().keyPressEvent(event)

    # =========================================================================
    # LOG
    # =========================================================================

    @Slot(str)
    def log_status(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.append(f"[{timestamp}] {message}")

    # =========================================================================
    # CONEXIÓN DE HARDWARE
    # =========================================================================

    def connect_piezo(self):
        try:
            serial = self.entry_piezo_serial.text().strip()
            self.log_status(f"Conectando a piezo {serial}...")
            DeviceManagerCLI.BuildDeviceList()
            self.piezo_device = BenchtopPrecisionPiezo.CreateBenchtopPiezo(serial)
            self.piezo_device.Connect(serial)

            channel = self.config.get('piezo_channel', 1)
            self.piezo_channel = self.piezo_device.GetChannel(channel)

            if not self.piezo_channel.IsSettingsInitialized():
                self.piezo_channel.WaitForSettingsInitialized(10000)

            self.piezo_channel.StartPolling(250)
            time.sleep(0.25)
            self.piezo_channel.EnableDevice()
            time.sleep(0.25)
            self.piezo_channel.SetPositionControlMode(Piezo.PiezoControlModeTypes.CloseLoop)
            time.sleep(0.25)

            self.piezo_connected = True
            self.piezo_status.setText("● Conectado")
            self.piezo_status.setStyleSheet("color: #a6e3a1;")
            self.btn_connect_piezo.setEnabled(False)
            self.btn_piezo_up.setEnabled(True)
            self.btn_piezo_down.setEnabled(True)
            self.btn_home.setEnabled(True)
            self.btn_goto.setEnabled(True)
            self.log_status("✓ Piezo conectado")
            self.update_ui_state()

        except Exception as e:
            self.log_status(f"✗ Error conectando piezo: {e}")
            QMessageBox.critical(self, "Error", f"No se pudo conectar al piezo:\n{e}")

    def connect_camera(self):
        try:
            serial = self.entry_camera_serial.text().strip()
            self.log_status(f"Conectando a cámara {serial}...")
            self.sdk = TLCameraSDK()
            available = self.sdk.discover_available_cameras()

            if len(available) < 1:
                raise Exception("No se detectó ninguna cámara")
            if serial not in available:
                self.log_status(f"Cámaras disponibles: {available}")
                raise Exception(f"Cámara {serial} no encontrada")

            self.camera = self.sdk.open_camera(serial)
            self.camera.exposure_time_us = int(self.entry_exposure.text())
            self.camera.frames_per_trigger_zero_for_unlimited = 0
            self.camera.image_poll_timeout_ms = 2000

            self.camera_connected = True
            self.camera_status.setText("● Conectada")
            self.camera_status.setStyleSheet("color: #a6e3a1;")
            self.btn_connect_camera.setEnabled(False)
            self.btn_set_exposure.setEnabled(True)
            self.btn_set_roi.setEnabled(True)
            self.btn_reset_roi.setEnabled(True)
            self.btn_preview.setEnabled(True)
            self.log_status("✓ Cámara conectada")
            self.log_status(f"  Resolución: {self.camera.image_width_pixels}x{self.camera.image_height_pixels}")
            self.update_ui_state()

        except Exception as e:
            self.log_status(f"✗ Error conectando cámara: {e}")
            QMessageBox.critical(self, "Error", f"No se pudo conectar a la cámara:\n{e}")

    def update_ui_state(self):
        both = self.piezo_connected and self.camera_connected
        self.btn_run_calib.setEnabled(both)
        self.btn_run_maxint.setEnabled(both)
        self.btn_set_setpoint.setEnabled(both)
        if both and self.calibrated:
            self.btn_start.setEnabled(True)

    # =========================================================================
    # CONTROL DEL PIEZO
    # =========================================================================

    def get_piezo_position(self):
        if not self.piezo_connected:
            return 0
        return float(str(self.piezo_channel.GetPosition()).replace(',', '.'))

    def set_piezo_position(self, pos_um):
        if not self.piezo_connected:
            return 0
        original = pos_um
        pos_um = np.clip(pos_um, self.pos_min_um, self.pos_max_um)
        if abs(original - pos_um) > 0.1 and self.running:
            self.log_status(f"⚠ Límite alcanzado: {pos_um:.1f} µm")
        self.piezo_channel.SetPosition(Decimal(pos_um))
        return pos_um

    def move_piezo(self, direction):
        if not self.piezo_connected:
            return
        try:
            step = float(self.entry_step.text())
            new_pos = self.set_piezo_position(self.get_piezo_position() + direction * step)
            self.log_status(f"Piezo → {new_pos:.2f} µm")
        except Exception as e:
            self.log_status(f"Error moviendo piezo: {e}")

    def piezo_home(self):
        if not self.piezo_connected:
            return
        home_pos = (self.pos_max_um + self.pos_min_um) / 2
        self.set_piezo_position(home_pos)
        self.log_status(f"Piezo → Home ({home_pos:.1f} µm)")

    def piezo_goto(self):
        if not self.piezo_connected:
            return
        try:
            new_pos = self.set_piezo_position(float(self.entry_goto.text()))
            self.log_status(f"Piezo → {new_pos:.2f} µm")
        except ValueError:
            QMessageBox.critical(self, "Error", "Posición inválida")

    def update_piezo_display(self):
        if self.piezo_connected:
            pos = self.get_piezo_position()
            self.current_piezo_pos = pos
            self.piezo_pos_label.setText(f"Posición: {pos:.2f} µm")

    def apply_piezo_limits(self):
        try:
            min_val = float(self.entry_piezo_min.text())
            max_val = float(self.entry_piezo_max.text())

            if min_val >= max_val:
                QMessageBox.critical(self, "Error", "El límite mínimo debe ser menor que el máximo")
                return

            if min_val < 0 or max_val > 450:
                reply = QMessageBox.question(self, "Advertencia",
                    f"Los límites están fuera del rango típico (0-450 µm).\n¿Continuar?")
                if reply != QMessageBox.Yes:
                    return

            self.pos_min_um = min_val
            self.pos_max_um = max_val
            self.limits_status_label.setText(f"Rango: {self.pos_min_um:.0f} - {self.pos_max_um:.0f} µm")
            home_pos = (self.pos_min_um + self.pos_max_um) / 2
            self.btn_home.setText(f"Home ({home_pos:.0f} µm)")
            self.entry_goto.setText(f"{home_pos:.0f}")

            if self.piezo_connected:
                current_pos = self.get_piezo_position()
                if current_pos < self.pos_min_um or current_pos > self.pos_max_um:
                    reply = QMessageBox.question(self, "Posición fuera de rango",
                        f"La posición actual ({current_pos:.1f} µm) está fuera de los nuevos límites.\n"
                        f"¿Mover a la posición central ({home_pos:.1f} µm)?")
                    if reply == QMessageBox.Yes:
                        self.set_piezo_position(home_pos)

            self.log_status(f"✓ Límites actualizados: {self.pos_min_um:.0f} - {self.pos_max_um:.0f} µm")

        except ValueError:
            QMessageBox.critical(self, "Error", "Valores inválidos. Ingresá números válidos.")
        except Exception as e:
            self.log_status(f"✗ Error aplicando límites: {e}")

    # =========================================================================
    # CONTROL DE CÁMARA
    # =========================================================================

    def set_exposure(self):
        if not self.camera_connected:
            return
        try:
            exp = int(self.entry_exposure.text())
            self.camera.exposure_time_us = exp
            self.log_status(f"Exposición → {exp} µs")
        except Exception as e:
            self.log_status(f"Error: {e}")

    def configure_roi(self):
        if not self.camera_connected:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Configurar ROI")
        dialog.setFixedWidth(280)
        layout = QVBoxLayout(dialog)

        title = QLabel("Configurar Región de Interés (ROI)")
        title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        grid = QGridLayout()
        fields = {}
        for i, (label, key) in enumerate([("X:", 'x'), ("Y:", 'y'), ("Ancho:", 'width'), ("Alto:", 'height')]):
            grid.addWidget(QLabel(label), i, 0, Qt.AlignRight)
            entry = QLineEdit(str(self.camera_roi.get(key, 0)))
            entry.setFixedWidth(80)
            grid.addWidget(entry, i, 1)
            fields[key] = entry

        layout.addLayout(grid)
        layout.addWidget(QLabel("(0 = full frame)"))

        btn_apply = QPushButton("Aplicar")
        btn_apply.setObjectName("btn_green")
        layout.addWidget(btn_apply)

        def apply_roi():
            try:
                self.camera_roi = {k: int(fields[k].text()) for k in fields}
                self.update_roi_label()
                self.log_status(f"ROI configurado: {self.camera_roi}")
                dialog.accept()
            except ValueError:
                QMessageBox.critical(dialog, "Error", "Valores inválidos")

        btn_apply.clicked.connect(apply_roi)
        dialog.exec()

    def reset_roi(self):
        self.camera_roi = {'x': 0, 'y': 0, 'width': 0, 'height': 0}
        self.update_roi_label()
        self.log_status("ROI reseteado a full frame")

    def update_roi_label(self):
        if self.camera_roi['width'] == 0:
            self.roi_label.setText("ROI: Full frame")
        else:
            r = self.camera_roi
            self.roi_label.setText(f"ROI: {r['x']},{r['y']}  {r['width']}×{r['height']}")

    def capture_image(self):
        if not self.camera_connected:
            return None

        self.camera.arm(2)
        self.camera.issue_software_trigger()
        frame = self.camera.get_pending_frame_or_null()

        if frame is None:
            self.camera.disarm()
            return None

        image_buffer = np.copy(frame.image_buffer)
        image = image_buffer.reshape(self.camera.image_height_pixels, self.camera.image_width_pixels)
        self.camera.disarm()

        if self.camera_roi['width'] > 0 and self.camera_roi['height'] > 0:
            x, y = self.camera_roi['x'], self.camera_roi['y']
            w, h = self.camera_roi['width'], self.camera_roi['height']
            image = image[y:y+h, x:x+w]

        return image

    # =========================================================================
    # ADQUISICIÓN COMPARTIDA
    # =========================================================================

    def _update_acq_interval_cache(self, text):
        """Actualiza el valor cacheado que usan los threads de fondo."""
        try:
            self._acq_interval_s = float(text) / 1000.0
        except ValueError:
            pass

    def _start_acquisition(self):
        if self._acq_thread is None or not self._acq_thread.is_alive():
            self.acquisition_running = True
            self._acq_thread = threading.Thread(target=self._acquisition_thread, daemon=True)
            self._acq_thread.start()

    def _stop_acquisition(self):
        if not self.running and not self.preview_active:
            self.acquisition_running = False

    def _acquisition_thread(self):
        while self.acquisition_running:
            interval_s = self._acq_interval_s  # leer caché, no el widget

            try:
                image = self.capture_image()
                if image is None:
                    time.sleep(interval_s)
                    continue

                max_idx = np.argmax(image)
                y_b, x_b = np.unravel_index(max_idx, image.shape)
                if image[y_b, x_b] >= self.saturation_threshold:
                    self.saturation_counter += 1
                    if self.saturation_counter >= 2:
                        self.saturation_counter = 0
                        self.sig_saturation.emit()
                        time.sleep(interval_s)
                        continue
                else:
                    self.saturation_counter = 0

                cx, cy, angle, intensity = calculate_psf_center(image)

                with self.image_lock:
                    self.current_image = image
                    self.current_cx = cx
                    self.current_cy = cy
                    self.current_angle = angle
                    self.current_intensity = intensity

                self.new_frame_event.set()

            except Exception as e:
                self.sig_log.emit(f"Error adquisición: {e}")

            time.sleep(interval_s)

    @Slot()
    def _handle_saturation(self):
        if self.preview_active:
            self.preview_active = False
            self.btn_preview.setText("▶  Iniciar Preview")
            self.preview_timer.stop()
            self._stop_acquisition()
        QMessageBox.warning(self, "¡Imagen Saturada!",
            f"Se detectó saturación (≥ {self.saturation_threshold} cts).\n"
            "El preview se pausó. Reducí la exposición antes de continuar.")

    # =========================================================================
    # PREVIEW
    # =========================================================================

    def toggle_preview(self):
        if self.preview_active:
            self.preview_active = False
            self.btn_preview.setText("▶  Iniciar Preview")
            self.preview_timer.stop()
            self._stop_acquisition()
        else:
            self.preview_active = True
            self.btn_preview.setText("■  Detener Preview")
            self._start_acquisition()
            interval_ms = max(50, int(self._acq_interval_s * 1000))
            self.preview_timer.start(interval_ms)

    @Slot()
    def update_preview(self):
        if not self.preview_active or not self.camera_connected:
            return

        try:
            with self.image_lock:
                image = self.current_image

            if image is None:
                return

            max_idx = np.argmax(image)
            y_bright, x_bright = np.unravel_index(max_idx, image.shape)
            max_val = image[y_bright, x_bright]

            self.fps_counter += 1
            if time.time() - self.fps_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_time = time.time()
                self.fps_label.setText(f"FPS: {self.current_fps}")

            h_orig, w_orig = image.shape[:2]
            scale = 500 / w_orig
            new_w, new_h = int(w_orig * scale), int(h_orig * scale)

            img_r = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img_n = cv2.normalize(img_r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_n, cv2.COLOR_GRAY2RGB)

            cv2.drawMarker(img_rgb,
                           (int(x_bright * scale), int(y_bright * scale)),
                           (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

            if self.calibrated and self.setpoint_x is not None:
                sx, sy = int(self.setpoint_x * scale), int(self.setpoint_y * scale)
                cv2.drawMarker(img_rgb, (sx, sy), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 20, 2)
                cv2.line(img_rgb, (sx, sy), (int(x_bright * scale), int(y_bright * scale)), (255, 255, 0), 1)

            self.image_label.setPixmap(self._array_to_pixmap(img_rgb))
            self.spot_info_label.setText(f"Max Px: {max_val} cts en ({x_bright}, {y_bright})")

        except Exception as e:
            self.log_status(f"Error en preview: {e}")

    # =========================================================================
    # CALIBRACIÓN
    # =========================================================================

    def load_calibration(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo de calibración", "", "JSON files (*.json)",
            options=QFileDialog.Option.DontUseNativeDialog)
        if not filename:
            return
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.slope_angle = data['metadata']['slope_angle_rad_per_um']
            self.slope_intensity = data['metadata']['slope_intensity_cts_per_um']
            self.update_sensitivity_display()
            self.calibrated = True
            self.update_ui_state()
            self.log_status("✓ Calibración JSON cargada exitosamente.")
            self.log_status(f"  Ángulo: {self.slope_angle:.5f} rad/µm | Intensidad: {self.slope_intensity:.2f} cts/µm")
        except KeyError as e:
            self.log_status(f"✗ Error: formato JSON incorrecto. Falta: {e}")
            QMessageBox.critical(self, "Error de Formato", f"El archivo JSON no tiene la estructura esperada.\nFalta: {e}")
        except Exception as e:
            self.log_status(f"✗ Error cargando calibración: {e}")
            QMessageBox.critical(self, "Error", str(e))

    def run_calibration(self):
        if not (self.camera_connected and self.piezo_connected):
            QMessageBox.critical(self, "Error", "Conecta todo el hardware primero")
            return
        try:
            calib_range = float(self.entry_calib_range.text())
            n_steps = int(self.entry_calib_steps.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Valores inválidos")
            return

        reply = QMessageBox.question(self, "Calibración",
            f"Scan de {calib_range} µm en {n_steps} pasos.\n¿Continuar?")
        if reply != QMessageBox.Yes:
            return

        self.calibrating = True
        self.btn_run_calib.setEnabled(False)
        self.log_status("Calibrando...")

        # Capturar valores de UI antes de lanzar el thread
        interval_s = self._acq_interval_s
        fmt = self.combo_calib_format.currentText()

        thread = threading.Thread(
            target=self._calibration_thread,
            args=(calib_range, n_steps, interval_s, fmt),
            daemon=True)
        thread.start()

    def _calibration_thread(self, calib_range, n_steps, interval_s, fmt):
        stabilization_s = max(0.3, interval_s)
        try:
            self._start_acquisition()
            start_pos = self.get_piezo_position()
            positions = np.linspace(start_pos - calib_range / 2,
                                    start_pos + calib_range / 2, n_steps)
            results = []

            for i, pos in enumerate(positions):
                self.set_piezo_position(pos)
                time.sleep(stabilization_s)
                self.new_frame_event.clear()
                got = self.new_frame_event.wait(timeout=interval_s * 3)
                if not got:
                    self.sig_log.emit(f"⚠ Timeout en paso {i+1}")
                    continue

                with self.image_lock:
                    cx, cy = self.current_cx, self.current_cy
                    angle, intensity = self.current_angle, self.current_intensity

                if cx is not None:
                    results.append({'z_um': pos, 'cx': cx, 'cy': cy,
                                    'angle': angle, 'intensity': intensity})
                self.sig_log.emit(f"Paso calib: {i+1}/{n_steps}")

            self.set_piezo_position(start_pos)

            if len(results) > 2:
                z_arr   = np.array([r['z_um']     for r in results])
                ang_arr = np.array([r['angle']    for r in results])
                int_arr = np.array([r['intensity'] for r in results])

                self.slope_angle    = np.polyfit(z_arr, ang_arr, 1)[0]
                self.slope_intensity = np.polyfit(z_arr, int_arr, 1)[0]

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                if fmt == 'CSV':
                    filename = f"calibracion_{timestamp}.csv"
                    with open(filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['piezo_um', 'cx_px', 'cy_px', 'angle_rad', 'intensity_counts'])
                        for r in results:
                            writer.writerow([round(r['z_um'], 3), round(r['cx'], 2), round(r['cy'], 2),
                                             round(r['angle'], 5), round(r['intensity'], 1)])
                else:
                    filename = f"calibracion_{timestamp}.json"
                    try:
                        calib_data = {
                            'metadata': {
                                'timestamp': timestamp,
                                'slope_angle_rad_per_um': float(self.slope_angle),
                                'slope_intensity_cts_per_um': float(self.slope_intensity),
                                'range_um': calib_range,
                                'steps': n_steps
                            },
                            'data': [
                                {'z_um': float(r['z_um']), 'cx': float(r['cx']),
                                 'cy': float(r['cy']), 'angle': float(r['angle']),
                                 'intensity': float(r['intensity'])}
                                for r in results
                            ]
                        }
                        with open(filename, 'w') as f:
                            json.dump(calib_data, f, indent=4)
                    except Exception as e:
                        self.sig_calib_finished.emit(False, f"✗ Error guardando JSON: {e}")
                        return

                self.calibrated = True
                self.sig_calib_data.emit({
                    'z_um': z_arr,
                    'angle': ang_arr,
                    'intensity': int_arr,
                    'slope_angle': float(self.slope_angle),
                    'slope_intensity': float(self.slope_intensity),
                })
                self.sig_calib_finished.emit(True, f"✓ Calibración guardada en {filename}")
            else:
                self.sig_calib_finished.emit(False, "✗ Falló: pocos puntos válidos")

        except Exception as e:
            self.sig_calib_finished.emit(False, f"✗ Error: {e}")
        finally:
            self.calibrating = False
            self._stop_acquisition()

    @Slot(object)
    def _show_calib_plot(self, calib_data: dict):
        dlg = CalibrationPlotDialog(calib_data, parent=self)
        dlg.show()   # no bloqueante: el usuario lo cierra cuando quiera

    @Slot(bool, str)
    def _on_calib_finished(self, success, message):
        self.log_status(message)
        if success:
            self.update_sensitivity_display()
            self.update_ui_state()
        self.btn_run_calib.setEnabled(True)

    def run_max_intensity_search(self):
        if not (self.camera_connected and self.piezo_connected):
            QMessageBox.critical(self, "Error", "Conecta todo el hardware primero")
            return
        try:
            search_range = float(self.entry_maxint_range.text())
            n_steps = int(self.entry_maxint_steps.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Valores inválidos")
            return

        reply = QMessageBox.question(self, "Búsqueda de Máximo",
            f"Scan de {search_range} µm en {n_steps} pasos.\n¿Continuar?")
        if reply != QMessageBox.Yes:
            return

        self.btn_run_maxint.setEnabled(False)
        self.lbl_maxint_result.setText("Buscando...")
        self.log_status("Buscando máximo de intensidad...")

        interval_s = self._acq_interval_s
        thread = threading.Thread(
            target=self._max_intensity_thread,
            args=(search_range, n_steps, interval_s),
            daemon=True)
        thread.start()

    def _max_intensity_thread(self, search_range, n_steps, interval_s):
        stabilization_s = max(0.3, interval_s)
        try:
            self._start_acquisition()
            start_pos = self.get_piezo_position()
            positions = np.linspace(start_pos - search_range / 2,
                                    start_pos + search_range / 2, n_steps)
            peak_values, valid_positions = [], []

            for i, pos in enumerate(positions):
                self.set_piezo_position(pos)
                time.sleep(stabilization_s)
                self.new_frame_event.clear()
                got = self.new_frame_event.wait(timeout=interval_s * 3)
                if not got:
                    self.sig_log.emit(f"⚠ Timeout en paso {i+1}")
                    continue

                with self.image_lock:
                    img = self.current_image.copy() if self.current_image is not None else None

                if img is not None:
                    peak_values.append(float(np.max(img)))
                    valid_positions.append(pos)
                self.sig_log.emit(f"Max int paso: {i+1}/{n_steps}")

            self.set_piezo_position(start_pos)

            if peak_values:
                best_idx = int(np.argmax(peak_values))
                best_pos = valid_positions[best_idx]
                best_val = peak_values[best_idx]
                self.sig_maxint_finished.emit(True, best_pos, best_val,
                    f"✓ Máximo de intensidad en {best_pos:.2f} µm ({best_val:.0f} cts)")
            else:
                self.sig_maxint_finished.emit(False, 0.0, 0.0, "✗ No se obtuvieron datos válidos")

        except Exception as e:
            self.sig_maxint_finished.emit(False, 0.0, 0.0, f"✗ Error: {e}")
        finally:
            self._stop_acquisition()

    @Slot(bool, float, float, str)
    def _on_maxint_finished(self, success, best_pos, best_val, message):
        self.log_status(message)
        if success:
            self.lbl_maxint_result.setText(f"Máximo en: {best_pos:.2f} µm  ({best_val:.0f} cts)")
        else:
            self.lbl_maxint_result.setText("Sin datos válidos")
        self.btn_run_maxint.setEnabled(True)

    def set_manual_sensitivity(self):
        try:
            ang_str = self.entry_sens_angle.text().strip()
            int_str = self.entry_sens_int.text().strip()
            if ang_str:
                self.slope_angle = float(ang_str)
            if int_str:
                self.slope_intensity = float(int_str)
            self.update_sensitivity_display()
            self.log_status(f"✓ Manual aplicado: Ángulo={self.slope_angle:.5f}, Int={self.slope_intensity:.2f}")
            self.calibrated = True
            self.update_ui_state()
        except ValueError:
            QMessageBox.critical(self, "Error", "Por favor ingresá valores numéricos válidos.")

    def update_sensitivity_display(self):
        ang_val = self.slope_angle if hasattr(self, 'slope_angle') else 0.0
        int_val = self.slope_intensity if hasattr(self, 'slope_intensity') else 0.0
        self.lbl_sens_angle.setText(f"Ángulo: {ang_val:.5f} rad/µm")
        self.lbl_sens_int.setText(f"Intensidad: {int_val:.2f} cts/µm")
        self.entry_sens_angle.setText(f"{ang_val:.5f}")
        self.entry_sens_int.setText(f"{int_val:.2f}")

    # =========================================================================
    # AUTOFOCO
    # =========================================================================

    def set_setpoint(self):
        if self.current_image is None:
            QMessageBox.critical(self, "Error", "No hay imagen en memoria. Iniciá el preview primero.")
            return

        self.log_status("Calculando setpoint exacto (PCA)...")
        cx, cy, angle, intensity = calculate_psf_center(self.current_image)

        if cx is None:
            QMessageBox.critical(self, "Error", "No se pudo detectar el spot para fijar el setpoint.")
            return

        self.setpoint_x = cx
        self.setpoint_y = cy
        self.setpoint_angle = angle
        self.setpoint_intensity = intensity
        self.calibrated = True

        angle_deg = np.degrees(angle) if angle is not None else 0
        self.setpoint_label.setText(f"Setpoint: ({cx:.1f}, {cy:.1f})")
        self.setpoint_angle_label.setText(f"Ángulo: {angle_deg:.2f}°  |  Intensidad: {intensity:.0f}")
        self.log_status(f"✓ Setpoint fijado: ({cx:.1f}, {cy:.1f}) | Ang={angle_deg:.2f}° | Int={intensity:.0f}")
        self.btn_start.setEnabled(True)

    def toggle_autofocus(self):
        if self.running:
            self.stop_autofocus()
        else:
            self.start_autofocus()

    def start_autofocus(self):
        if not self.calibrated:
            QMessageBox.critical(self, "Error", "Establecé el setpoint primero")
            return

        try:
            kp = float(self.entry_kp.text())
            ki = float(self.entry_ki.text())
            kd = float(self.entry_kd.text())
            self.pid = PIDController(kp, ki, kd)
            self.logger.clear()

            # Cachear valores para el thread (no leer widgets desde el thread)
            af_interval_s = self._acq_interval_s
            af_deadband_um = float(self.entry_deadband.text()) / 1000.0

            if self.preview_active:
                self.preview_active = False
                self.preview_timer.stop()
                self.btn_preview.setText("▶  Iniciar Preview")

            self.log_status(f"Iniciando autofoco..." if not self.chk_monitor_only.isChecked()
                            else "Iniciando MONITOREO (piezo fijo)...")
            self.log_status(f"  PID: Kp={kp}, Ki={ki}, Kd={kd} | Umbral: {af_deadband_um*1000:.0f} nm")

            self.running = True
            self.start_time = time.time()

            for d in [self.time_data, self.piezo_data, self.angle_data, self.intensity_data,
                      self.err_ang_data, self.err_int_data, self.z_err_unified_data,
                      self.z_err_ang_data, self.z_err_int_data]:
                d.clear()

            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_set_setpoint.setEnabled(False)

            self._start_acquisition()

            self.autofocus_thread = threading.Thread(
                target=self.autofocus_loop,
                args=(af_interval_s, af_deadband_um),
                daemon=True)
            self.autofocus_thread.start()

            interval_ms = max(50, int(af_interval_s * 1000))
            self.plots_timer.start(interval_ms)

        except Exception as e:
            self.log_status(f"✗ Error: {e}")

    def autofocus_loop(self, interval_s, deadband_um):
        while self.running:
            try:
                got_frame = self.new_frame_event.wait(timeout=interval_s * 2)
                self.new_frame_event.clear()

                if not got_frame:
                    continue

                with self.image_lock:
                    cx = self.current_cx
                    cy = self.current_cy
                    current_angle = self.current_angle
                    current_intensity = self.current_intensity

                if cx is None:
                    continue

                err_ang = current_angle - self.setpoint_angle
                err_int = current_intensity - self.setpoint_intensity

                z_err_ang = err_ang / self.slope_angle if self.slope_angle != 0 else 0
                z_err_int = err_int / self.slope_intensity if self.slope_intensity != 0 else 0

                if self.control_mode == 'angle':
                    unified_z_error = z_err_ang
                else:
                    unified_z_error = z_err_int

                control_signal_um = self.pid.update(unified_z_error, interval_s)

                current_pos = self.get_piezo_position()
                if not self.chk_monitor_only.isChecked() and abs(unified_z_error) > deadband_um:
                    new_pos = self.set_piezo_position(current_pos - control_signal_um)
                else:
                    new_pos = current_pos

                elapsed = time.time() - self.start_time
                self.time_data.append(elapsed)
                self.piezo_data.append(new_pos)
                self.angle_data.append(float(current_angle))
                self.intensity_data.append(float(current_intensity))
                self.err_ang_data.append(float(err_ang))
                self.err_int_data.append(float(err_int))
                self.z_err_unified_data.append(float(unified_z_error))
                self.z_err_ang_data.append(float(z_err_ang))
                self.z_err_int_data.append(float(z_err_int))

                self.logger.log(cx, cy, current_angle, current_intensity,
                                err_ang, err_int, unified_z_error,
                                control_signal_um, new_pos)

            except Exception as e:
                self.sig_log.emit(f"Error en loop: {e}")

    @Slot()
    def update_plots(self):
        if not self.running:
            return

        try:
            if len(self.time_data) > 1:
                t_list = list(self.time_data)

                self.line1.set_data(t_list, list(self.piezo_data))
                self.ax1.relim(); self.ax1.autoscale_view()

                self.line2.set_data(t_list, list(self.z_err_unified_data))
                try:
                    db_um = float(self.entry_deadband.text()) / 1000.0
                    self.line_db_up.set_ydata([db_um, db_um])
                    self.line_db_dn.set_ydata([-db_um, -db_um])
                except Exception:
                    pass
                self.ax2.relim(); self.ax2.autoscale_view()

                if self.control_mode == 'angle':
                    self.ax3.set_title("Error de Ángulo (crudo)", color='#cdd6f4', fontsize=8, pad=4)
                    self.ax3.set_ylabel("Δángulo (°)", color='#6c7086', fontsize=7)
                    self.line3a.set_data(t_list, np.degrees(list(self.err_ang_data)))
                    self.line3a.set_label("Δángulo (°)")
                else:
                    self.ax3.set_title("Error de Intensidad (crudo)", color='#cdd6f4', fontsize=8, pad=4)
                    self.ax3.set_ylabel("Δintensidad (cts)", color='#6c7086', fontsize=7)
                    self.line3a.set_data(t_list, list(self.err_int_data))
                    self.line3a.set_label("Δintensidad (cts)")

                self.line3a.set_color('#a6e3a1')
                self.line3b.set_data([], [])
                self.ax3.relim(); self.ax3.autoscale_view()
                self.ax3.legend(loc='upper right', fontsize=7,
                                facecolor='#313244', labelcolor='#cdd6f4')

                self.canvas.draw_idle()

            # Imagen en tiempo real
            with self.image_lock:
                image = self.current_image

            if image is not None:
                h_orig, w_orig = image.shape[:2]
                scale = 500 / w_orig
                new_w, new_h = int(w_orig * scale), int(h_orig * scale)

                img_r = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                img_n = cv2.normalize(img_r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                img_rgb = cv2.cvtColor(img_n, cv2.COLOR_GRAY2RGB)

                if self.current_cx is not None and self.current_cy is not None:
                    cx_s = int(self.current_cx * scale)
                    cy_s = int(self.current_cy * scale)
                    cv2.drawMarker(img_rgb, (cx_s, cy_s), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)

                    if self.current_angle is not None:
                        ang = self.current_angle
                        tip = (int(cx_s + 70 * np.cos(ang)), int(cy_s - 70 * np.sin(ang)))
                        cv2.arrowedLine(img_rgb, (cx_s, cy_s), tip, (0, 255, 0), 3, tipLength=0.25)

                    if self.setpoint_angle is not None:
                        sang = self.setpoint_angle
                        tip_s = (int(cx_s + 70 * np.cos(sang)), int(cy_s - 70 * np.sin(sang)))
                        cv2.arrowedLine(img_rgb, (cx_s, cy_s), tip_s, (255, 0, 0), 3, tipLength=0.25)

                    if self.setpoint_x is not None:
                        sx_s = int(self.setpoint_x * scale)
                        sy_s = int(self.setpoint_y * scale)
                        cv2.line(img_rgb, (sx_s, sy_s), (cx_s, cy_s), (100, 100, 100), 1)

                self.image_label.setPixmap(self._array_to_pixmap(img_rgb))

                ang_deg = np.degrees(self.current_angle) if self.current_angle is not None else 0
                delta_str = ""
                if self.setpoint_angle is not None:
                    delta_str = f"  Δáng: {np.degrees(self.current_angle - self.setpoint_angle):+.2f}°"
                self.spot_info_label.setText(
                    f"Spot: ({self.current_cx:.1f}, {self.current_cy:.1f})  |  "
                    f"Ang: {ang_deg:.1f}°{delta_str}  |  Int: {self.current_intensity:.0f}")

        except Exception:
            pass

    def stop_autofocus(self):
        self.running = False
        self.plots_timer.stop()
        self.log_status("Autofoco detenido")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_set_setpoint.setEnabled(True)
        self._stop_acquisition()

    # =========================================================================
    # GUARDAR DATOS
    # =========================================================================

    def save_log(self):
        if not self.logger.logs:
            QMessageBox.information(self, "Info", "No hay datos")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Guardar Log",
            f"autofocus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV files (*.csv)",
            options=QFileDialog.Option.DontUseNativeDialog)
        if filename:
            if not filename.lower().endswith('.csv'):
                filename += '.csv'
            try:
                self.logger.save(filename)
                self.log_status(f"✓ Log: {filename}")
            except Exception as e:
                self.log_status(f"✗ Error guardando log: {e}")
                QMessageBox.critical(self, "Error", f"No se pudo guardar el log:\n{e}")

    def save_error_data(self):
        if len(self.time_data) == 0:
            QMessageBox.information(self, "Info", "No hay datos en memoria para guardar.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Guardar Error Data",
            f"error_buffer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV files (*.csv)",
            options=QFileDialog.Option.DontUseNativeDialog)
        if filename:
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['time_s', 'piezo_um', 'angle_rad', 'intensity_cts',
                                     'err_angle_rad', 'err_intensity_cts', 'error_z_um'])
                    for t, p, a, i, ea, ei, ez in zip(
                            self.time_data, self.piezo_data, self.angle_data, self.intensity_data,
                            self.err_ang_data, self.err_int_data, self.z_err_unified_data):
                        writer.writerow([round(t,3), round(p,3), round(a,5), round(i,1),
                                         round(ea,5), round(ei,1), round(ez,4)])
                self.log_status(f"✓ Datos guardados en: {filename}")
            except Exception as e:
                self.log_status(f"✗ Error guardando datos: {e}")
                QMessageBox.critical(self, "Error", f"No se pudo guardar:\n{e}")

    def reload_config(self):
        self.config = self.load_config()
        self.log_status("Configuración recargada")

    # =========================================================================
    # UTILIDADES
    # =========================================================================

    def _array_to_pixmap(self, rgb_array):
        """Convierte un array numpy HxWx3 uint8 a QPixmap."""
        h, w, ch = rgb_array.shape
        qt_image = QImage(rgb_array.data, w, h, w * ch, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image.copy())

    def closeEvent(self, event):
        self.running = False
        self.preview_active = False
        self.calibrating = False
        self.acquisition_running = False
        self.piezo_timer.stop()
        self.preview_timer.stop()
        self.plots_timer.stop()
        try:
            if self.piezo_channel:
                self.piezo_channel.StopPolling()
                self.piezo_channel.DisableDevice()
            if self.piezo_device:
                self.piezo_device.Disconnect()
            if self.camera:
                self.camera.dispose()
            if self.sdk:
                self.sdk.dispose()
        except Exception:
            pass
        event.accept()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = AutofocusGUI()
    window.show()
    sys.exit(app.exec())
