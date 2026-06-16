"""
Live Stage Tracker con Phase Cross-Correlation y Failsafe (Random Walk)
Monitorea una carpeta en tiempo real, calcula el drift absoluto en XY basado 
en la proyección Max-Z inicial y exporta los datos a un archivo CSV.
Realiazdo en base a drift_correction_3d.py para un unico canal 
Autor: Tomás

Posible prompt para implementar: 
"Acá te paso un script llamado live_stage_tracker.py. 
Quiero integrarlo a esta GUI principal (código arriba). 
Por favor, transformá el bucle main() del tracker en una clase TrackerWorker(QThread)." 
"Conectalo al botón 'Iniciar Live Tracker' de la GUI. 
Usá Signal(str) para mandar los logs a mi QTextEdit y otra Signal(float, float, float) para el texto de los micrones.
Asegurate de que los parámetros como MAX_LIMIT_MICRONES y SIMULACION_OFFLINE los 
lea de los spinboxes de mi interfaz en el método __init__ del QThread."
"""

import os
import sys
import time
import glob
import csv
import re
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from skimage.registration import phase_cross_correlation
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor

# ──────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DEL TRACKER
# ──────────────────────────────────────────────────────────────────
ARCHIVO_CSV = "tracking_drift_stage.csv"    # Archivo de salida
ARCHIVO_LOG = "tracking_drift_log.txt"      # Archivo de log de eventos

FAILSAFE_OSCURIDAD = 0.4    # Umbral de caída de luz (0.4 = 40% del brillo original)
Z_STEP_1 = 5.0              # Primer paso del random walk en Z (micrones)
Z_STEP_2 = -10.0            # Segundo paso del random walk en Z (micrones)
UPSAMPLE = 1500              # Factor de precisión subpíxel para la correlación
USAR_ROI = True             # Habilitar selección de ROI en el primer frame
MAX_LIMIT_MICRONES = 20.0   # Límite máximo de excursión de la platina (absoluto)
SIMULACION_OFFLINE = True  # Poner en True SOLO para probar en la PC con TIFFs viejos
CAPAS_SLAB = 100             # Capas Z a incluir arriba y abajo del centro de máximo brillo

# ──────────────────────────────────────────────────────────────────
# FUNCIONES DE HARDWARE (API DEL MICROSCOPIO)
# ──────────────────────────────────────────────────────────────────
def mover_stage_xy(micrones_x: float, micrones_y: float):
    """
    [!] PLACEHOLDER PARA TU API [!]
    Agregá aquí tu comando para mover la platina en XY.
    Ejemplo: core.setXYPosition(x_actual + micrones_x, y_actual + micrones_y)
    """
    log_msg(f"    [#] [HARDWARE] Enviando comando al microscopio: Mover XY ({micrones_x:+.2f} µm, {micrones_y:+.2f} µm)")

def mover_stage_z(micrones: float):
    """
    [!] PLACEHOLDER PARA TU API [!]
    Reemplazá el print con la llamada real al hardware de tu stage.
    Ejemplo: core.setRelativePosition('ZDrive', micrones)
    """
    log_msg(f"    [#] [HARDWARE] Enviando comando al microscopio: Mover Z {micrones:+.1f} µm")

# ──────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────────────────────────
GUI_LOG_CALLBACK = None

def log_msg(msg: str):
    """Imprime en terminal y guarda en el archivo de log simultáneamente."""
    print(msg)
    with open(ARCHIVO_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    if GUI_LOG_CALLBACK:
        GUI_LOG_CALLBACK(msg)

def clean_bg(p):
    """Resta el fondo (percentil 1) para limpiar ruido sin distorsionar el contraste."""
    return np.clip(p - np.percentile(p, 1), 0, None)

def clamp_movement(requested_move, current_pos, max_limit):
    """Calcula el movimiento permitido sin exceder el límite absoluto de la platina."""
    new_pos = current_pos + requested_move
    if new_pos > max_limit:
        return max_limit - current_pos
    elif new_pos < -max_limit:
        return -max_limit - current_pos
    return requested_move

def obtener_limites_slab(z_center, z_dim, capas=10):
    """Asegura que el slab siempre tenga el mismo tamaño (2*capas + 1) para la correlación."""
    if z_dim <= 2 * capas + 1:
        return 0, z_dim
    c = max(capas, min(z_center, z_dim - capas - 1))
    return c - capas, c + capas + 1

def leer_tiff_seguro(filepath, retries=20, delay=1.0):
    """Intenta leer el TIFF. Es útil porque el archivo puede estar bloqueado 
    mientras el microscopio aún lo está escribiendo en el disco."""
    for i in range(retries):
        try:
            return tifffile.imread(filepath).astype(np.float32)
        except Exception:
            if i > 0:
                log_msg(f"    [~] Archivo ocupado por el sistema, esperando... ({i+1}/{retries})")
            time.sleep(delay)
    raise IOError(f"No se pudo leer el archivo de forma segura tras {retries} intentos: {filepath}")

# ──────────────────────────────────────────────────────────────────
# BUCLE PRINCIPAL (WATCHDOG)
# ──────────────────────────────────────────────────────────────────
def main(carpeta_deskews):
    global GUI_LOG_CALLBACK
    global ARCHIVO_CSV, ARCHIVO_LOG
    
    # Forzar a que los logs se guarden en la carpeta de este script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ARCHIVO_CSV = os.path.join(script_dir, "tracking_drift_stage.csv")
    ARCHIVO_LOG = os.path.join(script_dir, "tracking_drift_log.txt")
    
    # Ventana flotante para frenar la ejecución y ver el log en vivo
    control_window = QWidget()
    control_window.setWindowTitle("Live Tracker")
    control_window.resize(600, 400)
    layout = QVBoxLayout(control_window)
    
    lbl_drift = QLabel("Drift Acumulado: X: +0.00 µm | Y: +0.00 µm | Z: +0.00 µm")
    lbl_drift.setStyleSheet("font-size: 15px; font-weight: bold; color: #569cd6; padding: 5px;")
    lbl_drift.setAlignment(Qt.AlignCenter)
    layout.addWidget(lbl_drift)

    log_console = QTextEdit()
    log_console.setReadOnly(True)
    log_console.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas, 'Courier New', monospace; font-size: 12px;")
    layout.addWidget(log_console)
    
    def append_log(msg):
        log_console.append(msg)
        # Hacer auto-scroll hacia la última línea
        cursor = log_console.textCursor()
        cursor.movePosition(QTextCursor.End)
        log_console.setTextCursor(cursor)
        QApplication.processEvents() # Refresca visualmente al instante
        
    GUI_LOG_CALLBACK = append_log
    
    btn_stop = QPushButton("[X] Detener Tracker")
    btn_stop.setStyleSheet("background-color: #c0392b; color: white; font-size: 14px; font-weight: bold; padding: 15px;")
    layout.addWidget(btn_stop)
    
    estado_ejecucion = {"corriendo": True}
    def detener():
        estado_ejecucion["corriendo"] = False
        btn_stop.setText("Deteniendo...")
        btn_stop.setEnabled(False)
    btn_stop.clicked.connect(detener)
    control_window.show()

    # Inicializar Log
    with open(ARCHIVO_LOG, mode="w", encoding="utf-8") as f:
        f.write(f"=== Registro de Stage Tracker iniciado el {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        
    log_msg(f"=== Iniciando Stage Tracker ===")
    log_msg(f"Monitoreando carpeta: {carpeta_deskews}")
    log_msg(f"Exportando datos a : {ARCHIVO_CSV}")
    log_msg(f"Exportando log a   : {ARCHIVO_LOG}\n")
    
    # Inicializar CSV
    with open(ARCHIVO_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Archivo", "Estado", "Brillo", "dx", "dy", "dz", "Accion"])

    archivos_procesados = set()
    prev_xy, prev_xz, prev_yz = None, None, None
    z_min_prev = 0
    acum_dx, acum_dy, acum_dz = 0.0, 0.0, 0.0
    last_good_vmax = None
    
    # Máquina de estados para el Random Walk
    # 0 = OK, 1 = Ejecutado 5µm, 2 = Ejecutado -10µm
    recovery_step = 0 
    frame_idx = 0
    
    # Coordenadas absolutas acumuladas para validar los límites de seguridad
    stage_x, stage_y, stage_z = 0.0, 0.0, 0.0
    xmin, xmax, ymin, ymax = None, None, None, None

    try:
        while estado_ejecucion["corriendo"]:
            QApplication.processEvents()
            
            # Buscar archivos y ordenarlos numéricamente por el número al final de su nombre
            archivos_actuales = sorted(glob.glob(os.path.join(carpeta_deskews, "*.tif")), 
                                       key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[-1]) if re.findall(r"\d+", os.path.basename(x)) else 0)
            
            archivos_nuevos = [f for f in archivos_actuales if f not in archivos_procesados]
            
            for archivo in archivos_nuevos:
                # Salir inmediatamente si el usuario apretó detener
                if not estado_ejecucion["corriendo"]:
                    break
                    
                log_msg(f"[{time.strftime('%H:%M:%S')}] Nuevo deskew detectado: {os.path.basename(archivo)}")
                
                vol = leer_tiff_seguro(archivo)
                vmax_actual = np.percentile(vol, 99.9)
                
                # 1. Definir Referencia (t=0)
                if prev_xy is None:
                    if USAR_ROI:
                        mip_t0 = np.max(vol, axis=0)
                        Z_dim, Y_dim, X_dim = vol.shape
                        roi_coords = [0, X_dim, 0, Y_dim]
                        
                        def onselect(eclick, erelease):
                            roi_coords[0] = int(min(eclick.xdata, erelease.xdata))
                            roi_coords[1] = int(max(eclick.xdata, erelease.xdata))
                            roi_coords[2] = int(min(eclick.ydata, erelease.ydata))
                            roi_coords[3] = int(max(eclick.ydata, erelease.ydata))

                        log_msg("    >> Abriendo ventana para selección de ROI...")
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.imshow(mip_t0, cmap='gray')
                        ax.set_title("Seleccioná el ROI en XY y luego cerrá la ventana")
                        rs = RectangleSelector(ax, onselect, useblit=True, button=[1], interactive=True)
                        
                        plt.show(block=False)
                        while plt.fignum_exists(fig.number) and estado_ejecucion["corriendo"]:
                            QApplication.processEvents()
                            time.sleep(0.05)
                            
                        if not estado_ejecucion["corriendo"]:
                            plt.close(fig)
                            break
                        
                        xmin, xmax, ymin, ymax = roi_coords
                        xmin, xmax = max(0, xmin), min(X_dim, xmax)
                        ymin, ymax = max(0, ymin), min(Y_dim, ymax)
                        if (xmax - xmin) < 5 or (ymax - ymin) < 5:
                            log_msg("    [!] ROI inválido. Usando imagen completa.")
                            xmin, xmax, ymin, ymax = 0, X_dim, 0, Y_dim
                        log_msg(f"    [v] ROI seleccionado: X[{xmin}:{xmax}], Y[{ymin}:{ymax}]")
                    else:
                        Z_dim, Y_dim, X_dim = vol.shape
                        xmin, xmax, ymin, ymax = 0, X_dim, 0, Y_dim
                        
                    subvol = vol[:, ymin:ymax, xmin:xmax]
                    
                    # Seguimiento Grueso (Coarse): Encontrar Z de máximo brillo
                    ref_z_center = int(np.argmax(np.max(subvol, axis=(1, 2))))
                    z_min_prev, z_max_prev = obtener_limites_slab(ref_z_center, Z_dim, CAPAS_SLAB)
                    subvol_slab = subvol[z_min_prev:z_max_prev, :, :]
                    
                    # PROYECCIÓN XY FIJA: Usa el volumen completo para máxima estabilidad
                    prev_xy = np.max(subvol, axis=0)
                    prev_xz = np.max(subvol_slab, axis=1)
                    prev_yz = np.max(subvol_slab, axis=2)
                    last_good_vmax = vmax_actual
                    log_msg(f"    [v] Referencia anclada. Z-Centro: {ref_z_center}, Brillo: {vmax_actual:.1f}")
                    
                    with open(ARCHIVO_CSV, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([frame_idx, os.path.basename(archivo), "Referencia", vmax_actual, 0, 0, 0, "Anclaje"])
                    
                    archivos_procesados.add(archivo)
                    frame_idx += 1
                    continue
                
                # 2. Control de Failsafe y Random Walk
                if vmax_actual < last_good_vmax * FAILSAFE_OSCURIDAD:
                    log_msg(f"    [!] Brillo crítico detectado ({vmax_actual:.1f} vs ideal {last_good_vmax:.1f}).")
                    
                    if recovery_step == 0:
                        paso = clamp_movement(Z_STEP_1, stage_z, MAX_LIMIT_MICRONES)
                        mover_stage_z(paso)
                        stage_z += paso
                        recovery_step = 1
                        accion = f"Failsafe 1: {paso:+.1f} µm"
                    elif recovery_step == 1:
                        paso = clamp_movement(Z_STEP_2, stage_z, MAX_LIMIT_MICRONES)
                        mover_stage_z(paso)
                        stage_z += paso
                        recovery_step = 2
                        accion = f"Failsafe 2: {paso:+.1f} µm"
                    else:
                        # Si ya probamos ambas posiciones y no se recupera, tiramos el error
                        error_msg = f"Fallo irrecuperable. La muestra no apareció tras moverse {Z_STEP_1}µm y {Z_STEP_2}µm."
                        log_msg(f"    [!] {error_msg}")
                        QMessageBox.critical(None, "Error Crítico", error_msg)
                        raise RuntimeError(error_msg)
                    
                    with open(ARCHIVO_CSV, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([frame_idx, os.path.basename(archivo), "Failsafe", vmax_actual, "", "", "", accion])
                else:
                    # 3. Brillo OK -> Tracking Absoluto respecto a t=0 en 3D
                    recovery_step = 0
                    last_good_vmax = vmax_actual
                    
                    subvol_curr = vol[:, ymin:ymax, xmin:xmax]
                    
                    curr_z_center = int(np.argmax(np.max(subvol_curr, axis=(1, 2))))
                    z_min_curr, z_max_curr = obtener_limites_slab(curr_z_center, vol.shape[0], CAPAS_SLAB)
                    subvol_slab_curr = subvol_curr[z_min_curr:z_max_curr, :, :]
                    
                    # PROYECCIÓN XY FIJA: Usa el volumen completo para máxima estabilidad
                    curr_xy = np.max(subvol_curr, axis=0)
                    curr_xz = np.max(subvol_slab_curr, axis=1)
                    curr_yz = np.max(subvol_slab_curr, axis=2)
                    
                    shift_xy, err_xy, _ = phase_cross_correlation(clean_bg(prev_xy), clean_bg(curr_xy), upsample_factor=UPSAMPLE)
                    shift_xz, err_xz, _ = phase_cross_correlation(clean_bg(prev_xz), clean_bg(curr_xz), upsample_factor=UPSAMPLE)
                    shift_yz, err_yz, _ = phase_cross_correlation(clean_bg(prev_yz), clean_bg(curr_yz), upsample_factor=UPSAMPLE)
                    
                    rel_dy, rel_dx = shift_xy[0], shift_xy[1]
                    
                    weight_xz = 1.0 / (err_xz + 1e-5)
                    weight_yz = 1.0 / (err_yz + 1e-5)
                    rel_residual_dz = (shift_xz[0] * weight_xz + shift_yz[0] * weight_yz) / (weight_xz + weight_yz)
                    
                    # Diferencia de Slab
                    rel_coarse_dz = z_min_prev - z_min_curr
                    rel_dz = rel_coarse_dz + rel_residual_dz
                    
                    # ACUMULACIÓN PROGRESIVA (Idéntico a drift_correction_3d.py)
                    acum_dx += rel_dx
                    acum_dy += rel_dy
                    acum_dz += rel_dz
                    
                    # Actualizar memoria del frame previo
                    prev_xy = curr_xy
                    prev_xz = curr_xz
                    prev_yz = curr_yz
                    z_min_prev = z_min_curr
                    
                    raw_x = acum_dx * 0.127
                    raw_y = acum_dy * 0.127
                    raw_z = acum_dz * (1/4)
                    
                    if SIMULACION_OFFLINE:
                        paso_x = raw_x - stage_x
                        paso_y = raw_y - stage_y
                        paso_z = raw_z - stage_z
                    else:
                        # Si estamos conectados al hardware real
                        paso_x = rel_dx * 0.127
                        paso_y = rel_dy * 0.127
                        paso_z = rel_dz * (1/4)
                        
                    x_micrones = clamp_movement(paso_x, stage_x, MAX_LIMIT_MICRONES)
                    y_micrones = clamp_movement(paso_y, stage_y, MAX_LIMIT_MICRONES)
                    z_micrones = clamp_movement(paso_z, stage_z, MAX_LIMIT_MICRONES)
                    
                    if paso_x != x_micrones or paso_y != y_micrones or paso_z != z_micrones:
                        log_msg("    [!] Límite de seguridad alcanzado. El movimiento ha sido truncado.")
                        QMessageBox.warning(None, "Límite de Seguridad", f"Límite de seguridad de la platina ({MAX_LIMIT_MICRONES} µm) alcanzado.\n\nEl movimiento ha sido truncado preventivamente para evitar daños.")
                        
                    stage_x += x_micrones
                    stage_y += y_micrones
                    stage_z += z_micrones
                    
                    lbl_drift.setText(f"Drift Acumulado: X: {stage_x:+.2f} µm | Y: {stage_y:+.2f} µm | Z: {stage_z:+.2f} µm")
                    
                    log_msg(f"    [v] Tracking exitoso. Drift Acumulado: XY({acum_dx:+.2f}, {acum_dy:+.2f}) px | Z={acum_dz:+.2f} px")
                    
                    # Retroalimentación en vivo (Lazo Cerrado)
                    if not SIMULACION_OFFLINE:
                        mover_stage_xy(x_micrones, y_micrones)
                        mover_stage_z(z_micrones)
                    
                    accion_str = f"Corr XY({x_micrones:+.2f}, {y_micrones:+.2f}) Z({z_micrones:+.2f})"
                    with open(ARCHIVO_CSV, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([frame_idx, os.path.basename(archivo), "OK", vmax_actual, acum_dx, acum_dy, acum_dz, accion_str])
                
                archivos_procesados.add(archivo)
                frame_idx += 1
                
            # Esperar un segundo de forma no bloqueante para la GUI
            for _ in range(10):
                if not estado_ejecucion["corriendo"]: break
                QApplication.processEvents()
                time.sleep(0.1)
            
    except KeyboardInterrupt:
        log_msg("\n[!] Watchdog detenido por el usuario.")
        
    control_window.close()
    log_msg("\n[v] Watchdog finalizado correctamente.")

if __name__ == "__main__":
    # Creamos la app de PySide6 de forma segura (por si ya hay otra corriendo)
    app = QApplication.instance() or QApplication(sys.argv)
    
    carpeta_seleccionada = QFileDialog.getExistingDirectory(None, "Seleccionar Carpeta de Deskews a monitorear", "")
    
    if not carpeta_seleccionada:
        print("\n[!] No se seleccionó ninguna carpeta. Operación cancelada.")
        sys.exit(0)
        
    main(carpeta_seleccionada)
