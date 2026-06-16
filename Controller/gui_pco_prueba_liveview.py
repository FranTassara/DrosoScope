import sys
import time
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QDoubleSpinBox, 
                               QGroupBox, QMessageBox)
from PySide6.QtCore import QThread, Signal, Slot, Qt
from PySide6.QtGui import QImage, QPixmap

# Importamos tu controlador (debe estar en la misma carpeta como C_pco_edge42.py)
try:
    from C_pco_edge42 import Camera, CameraError
except ImportError:
    print("Error: No se encontró 'C_pco_edge42.py'. Asegúrate de que esté en la misma carpeta.")
    sys.exit(1)

class CameraWorker(QThread):
    """
    Hilo dedicado a la adquisición de imágenes.
    Esto evita que la GUI se congele mientras espera datos de la cámara.
    """
    image_ready = Signal(np.ndarray)
    error_occurred = Signal(str)

    def __init__(self, camera_instance):
        super().__init__()
        self.camera = camera_instance
        self.is_running = False
        self.target_fps = 30
        self._buffer = None # Reutilizaremos este buffer para no saturar la memoria

    def prepare_buffer(self):
        # Pre-asignamos memoria basada en el ROI actual para eficiencia
        h, w = self.camera.height_px, self.camera.width_px
        self._buffer = np.zeros((1, h, w), dtype='uint16')

    def run(self):
        self.is_running = True
        
        while self.is_running:
            t_start = time.perf_counter()
            
            try:
                # Usamos la función existente de tu controlador.
                # Pedimos solo 1 imagen, pero reutilizamos la memoria asignada.
                # software_trigger=False asume que la cámara está en 'auto' o recibiendo trigger externo
                self.camera.record_to_memory(allocated_memory=self._buffer, software_trigger=False)
                
                # Emitimos la imagen (extraemos la dimensión 0 ya que shape es (1, h, w))
                self.image_ready.emit(self._buffer[0])
                
            except Exception as e:
                self.error_occurred.emit(str(e))
                self.is_running = False
                break

            # Control simple de Frame Rate (espera si fuimos muy rápido)
            elapsed = time.perf_counter() - t_start
            wait_time = (1.0 / self.target_fps) - elapsed
            if wait_time > 0:
                time.sleep(wait_time)

    def stop(self):
        self.is_running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCO.edge 4.2 Controller")
        self.resize(800, 700)

        self.camera = None
        self.worker = None

        # --- Layout Principal ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 1. Área de Imagen
        self.lbl_image = QLabel("No Camera Connected")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background-color: #222; color: #888; font-size: 16px;")
        self.lbl_image.setMinimumSize(640, 480)
        main_layout.addWidget(self.lbl_image, stretch=1)

        # 2. Controles
        controls_layout = QHBoxLayout()
        main_layout.addLayout(controls_layout)

        # Grupo: Conexión
        grp_conn = QGroupBox("Conexión")
        layout_conn = QVBoxLayout()
        self.btn_connect = QPushButton("Conectar Cámara")
        self.btn_connect.clicked.connect(self.connect_camera)
        self.btn_disconnect = QPushButton("Desconectar")
        self.btn_disconnect.clicked.connect(self.disconnect_camera)
        self.btn_disconnect.setEnabled(False)
        layout_conn.addWidget(self.btn_connect)
        layout_conn.addWidget(self.btn_disconnect)
        grp_conn.setLayout(layout_conn)
        controls_layout.addWidget(grp_conn)

        # Grupo: Configuración (CORREGIDO: Usamos self.grp_config)
        self.grp_config = QGroupBox("Configuración") 
        layout_config = QVBoxLayout()
        
        # Exposure Time
        layout_exp = QHBoxLayout()
        layout_exp.addWidget(QLabel("Exposure (ms):"))
        self.spin_exposure = QDoubleSpinBox()
        self.spin_exposure.setRange(0.1, 1000.0) # 0.1ms a 1s
        self.spin_exposure.setValue(10.0)
        self.spin_exposure.setSuffix(" ms")
        self.btn_set_params = QPushButton("Aplicar")
        self.btn_set_params.clicked.connect(self.apply_parameters)
        layout_exp.addWidget(self.spin_exposure)
        layout_exp.addWidget(self.btn_set_params)
        layout_config.addLayout(layout_exp)

        # Target FPS (Limitador de software)
        layout_fps = QHBoxLayout()
        layout_fps.addWidget(QLabel("Target FPS:"))
        self.spin_fps = QDoubleSpinBox()
        self.spin_fps.setRange(1, 100)
        self.spin_fps.setValue(30)
        self.spin_fps.valueChanged.connect(self.update_fps_limit)
        layout_fps.addWidget(self.spin_fps)
        layout_config.addLayout(layout_fps)
        
        self.grp_config.setLayout(layout_config)
        controls_layout.addWidget(self.grp_config)

        # Grupo: Adquisición
        grp_acq = QGroupBox("Live View")
        layout_acq = QVBoxLayout()
        self.btn_start = QPushButton("Start Live")
        self.btn_start.clicked.connect(self.start_live)
        self.btn_start.setEnabled(False)
        self.btn_stop = QPushButton("Stop Live")
        self.btn_stop.clicked.connect(self.stop_live)
        self.btn_stop.setEnabled(False)
        layout_acq.addWidget(self.btn_start)
        layout_acq.addWidget(self.btn_stop)
        grp_acq.setLayout(layout_acq)
        controls_layout.addWidget(grp_acq)

        # Desactivar controles hasta conectar (CORREGIDO)
        self.grp_config.setEnabled(False)

    def connect_camera(self):
        try:
            self.camera = Camera() # Instancia tu clase
            self.lbl_image.setText("Cámara Conectada. Lista.")
            
            # Habilitar UI
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
            self.grp_config.setEnabled(True) # CORREGIDO: Uso directo de la variable
            self.btn_start.setEnabled(True)
            
            # Aplicar configuración inicial
            self.apply_parameters()
            
        except Exception as e:
            QMessageBox.critical(self, "Error de Conexión", str(e))

    def disconnect_camera(self):
        if self.worker and self.worker.isRunning():
            self.stop_live()
            
        if self.camera:
            self.camera.close()
            self.camera = None
            
        self.lbl_image.setText("Desconectado")
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.grp_config.setEnabled(False) # CORREGIDO: Uso directo de la variable

    def apply_parameters(self):
        if not self.camera: return

        # Si estamos en vivo, hay que pausar, configurar y reanudar
        was_running = False
        if self.worker and self.worker.isRunning():
            self.stop_live()
            was_running = True

        exp_ms = self.spin_exposure.value()
        exp_us = int(exp_ms * 1000)

        try:
            # Usamos tu método apply_settings
            # Configuramos num_buffers=4 para tener un buffer circular fluido
            self.camera.apply_settings(
                exposure_us=exp_us,
                trigger="auto", # 'auto' para free-run en liveview
                num_buffers=4 
            )
            print(f"Parámetros aplicados: {exp_ms}ms")
        except Exception as e:
            QMessageBox.warning(self, "Error Config", str(e))

        if was_running:
            self.start_live()

    def update_fps_limit(self, val):
        if self.worker:
            self.worker.target_fps = val

    def start_live(self):
        if not self.camera: return
        
        # Crear e iniciar el hilo
        self.worker = CameraWorker(self.camera)
        self.worker.prepare_buffer()
        self.worker.target_fps = self.spin_fps.value()
        self.worker.image_ready.connect(self.update_display)
        self.worker.error_occurred.connect(self.handle_worker_error)
        self.worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.spin_exposure.setEnabled(False) # Bloquear exposure en live (requiere re-armado)
        self.btn_set_params.setEnabled(False)

    def stop_live(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.spin_exposure.setEnabled(True)
        self.btn_set_params.setEnabled(True)

    @Slot(str)
    def handle_worker_error(self, msg):
        self.stop_live()
        QMessageBox.critical(self, "Error de Adquisición", msg)

    @Slot(np.ndarray)
    def update_display(self, img_data):
        """
        Convierte el array numpy (uint16) a QImage y lo muestra.
        NOTA: Esto es lento para 4K, pero sirve para visualizar.
        """
        # 1. Normalización simple para visualización (16-bit -> 8-bit)
        # Esto auto-escala el brillo basado en el min/max de la imagen actual
        min_val = np.min(img_data)
        max_val = np.max(img_data)
        
        if max_val == min_val:
            disp_data = np.zeros_like(img_data, dtype=np.uint8)
        else:
            disp_data = ((img_data - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)

        h, w = disp_data.shape
        bytes_per_line = w
        
        # Crear QImage desde datos
        q_img = QImage(disp_data.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        
        # Escalar al tamaño del label (manteniendo aspect ratio)
        pixmap = QPixmap.fromImage(q_img)
        self.lbl_image.setPixmap(pixmap.scaled(self.lbl_image.size(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.disconnect_camera()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())