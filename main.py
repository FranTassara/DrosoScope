"""
OPM Microscopy Control Application

Multichannel acquisition with temporal scheduler for interleaved
volume imaging.
"""

import sys
import os

import qdarkstyle
import numpy as np
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PyQt5.QtCore import QThread
from pyqtgraph import RectROI, SignalProxy
from pylablib.devices.Thorlabs.TLCamera import ThorlabsTLCamera

from controllers.laser_488nm import OxxiusLaser488
from controllers.laser_561nm import OxxiusLaser561
from controllers.stage import AsiStage
from controllers.filter_wheel import FilterWheel
from controllers.flipper import MFF101
from controllers import camera_pco
from controllers import daq

from config import HARDWARE_CONFIG
from acquisition import (LiveViewThread, SerialDeviceInitThread,
                         FilterWheelInitThread, MultichannelSchedulerWorker)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('OPM Control App')

        base_dir = os.path.dirname(os.path.abspath(__file__))
        ui_file = os.path.join(base_dir, 'ui', 'MainApp.ui')
        uic.loadUi(ui_file, self)

        # =====================================================================
        # Laser 488
        # =====================================================================
        self.laser488 = OxxiusLaser488(HARDWARE_CONFIG['laser_488_port'])
        self.pushButton_connect_488.clicked.connect(self.toggle_laser_488)
        self.pushButton_cmd_488.clicked.connect(self.command_488)
        self.pushButton_on_488.clicked.connect(lambda: self.laser488.set_laser_control("ON"))
        self.pushButton_off_488.clicked.connect(lambda: self.laser488.set_laser_control("OFF"))
        self.pushButton_setP_488.clicked.connect(self.set_power_488)
        self.pushButton_setC_488.clicked.connect(self.set_current_488)
        self.pushButton_updateStat_488.clicked.connect(self.refresh_488)
        self.radioButton_cteP_488.toggled.connect(self.set_constant_power_mode_488)
        self.radioButton_cteC_488.toggled.connect(self.set_constant_current_mode_488)
        self.laser488_connected = False

        # =====================================================================
        # Laser 561
        # =====================================================================
        self.laser561 = OxxiusLaser561(HARDWARE_CONFIG['laser_561_port'])
        self.pushButton_connect_561.clicked.connect(self.toggle_laser_561)
        self.pushButton_on_561.clicked.connect(lambda: self.laser561.set_laser_control("ON"))
        self.pushButton_off_561.clicked.connect(self._confirm_laser_561_off)
        self.pushButton_SHo_561.clicked.connect(lambda: self.laser561.set_shutter("OPEN"))
        self.pushButton_SHc_561.clicked.connect(lambda: self.laser561.set_shutter("CLOSE"))
        self.pushButton_cmd_561.clicked.connect(self.command_561)
        self.laser561_connected = False

        # =====================================================================
        # Stage ASI
        # =====================================================================
        self.stage = AsiStage(HARDWARE_CONFIG['stage_port'])
        self.pushButton_connect_Stage.clicked.connect(self.connect_stage)
        self.pushButton_halt.clicked.connect(self.stage.halt)
        self.pushButton_zero.clicked.connect(self.stage.zero)
        self.pushButton_relative_posX.clicked.connect(
            lambda: self.stage.move_relative_axis("X", int(self.doubleSpinBox_X.value() * 10)))
        self.pushButton_relative_negX.clicked.connect(
            lambda: self.stage.move_relative_axis("X", -int(self.doubleSpinBox_X.value() * 10)))
        self.pushButton_absoluteX.clicked.connect(
            lambda: self.stage.move_absolute_axis("X", int(self.doubleSpinBox_X.value() * 10)))
        self.pushButton_relative_posY.clicked.connect(
            lambda: self.stage.move_relative_axis("Y", int(self.doubleSpinBox_Y.value() * 10)))
        self.pushButton_relative_negY.clicked.connect(
            lambda: self.stage.move_relative_axis("Y", -int(self.doubleSpinBox_Y.value() * 10)))
        self.pushButton_absoluteY.clicked.connect(
            lambda: self.stage.move_absolute_axis("Y", int(self.doubleSpinBox_Y.value() * 10)))
        self.pushButton_relative_posZ.clicked.connect(
            lambda: self.stage.move_relative_axis("Z", int(self.doubleSpinBox_Z.value() * 10)))
        self.pushButton_relative_negZ.clicked.connect(
            lambda: self.stage.move_relative_axis("Z", -int(self.doubleSpinBox_Z.value() * 10)))
        self.pushButton_absoluteZ.clicked.connect(
            lambda: self.stage.move_absolute_axis("Z", int(self.doubleSpinBox_Z.value() * 10)))
        self.pushButton_updateStage.clicked.connect(self.refresh_stage)
        self.pushButton_relative_posXprima.clicked.connect(lambda: self.move_relative_xprime(1))
        self.pushButton_relative_negXprima.clicked.connect(lambda: self.move_relative_xprime(-1))
        self.pushButton_relative_posZprima.clicked.connect(lambda: self.move_relative_zprime(1))
        self.pushButton_relative_negZprima.clicked.connect(lambda: self.move_relative_zprime(-1))
        self.stage_connected = False

        # =====================================================================
        # Camera PCO
        # =====================================================================
        self.pushButton_connect_PCO.clicked.connect(self.connect_camera_pco)
        self.pushButton_disconnect_PCO.clicked.connect(self.disconnect_camera_pco)
        self.pushButton_LiveView_PCO.clicked.connect(self.toggle_live_view_pco)
        self.pushButton_createROI_PCO.clicked.connect(self.toggle_roi_mode_pco)
        self.pushButton_Snap_PCO.clicked.connect(self.snap_image_pco)
        self.camera = None
        self.camera_connected = False
        self.live_thread = None
        self.roi_enabled = False
        self.roi_rect = None
        self._image_initialized = False
        # Connect spinbox signals once (they check roi_enabled internally)
        self.spinBox_hstart.editingFinished.connect(self._update_roi_from_spinboxes_pco)
        self.spinBox_hend.editingFinished.connect(self._update_roi_from_spinboxes_pco)
        self.spinBox_vstart.editingFinished.connect(self._update_roi_from_spinboxes_pco)
        self.spinBox_vend.editingFinished.connect(self._update_roi_from_spinboxes_pco)

        # =====================================================================
        # Camera Thorlabs
        # =====================================================================
        self.pushButton_connect_TL.clicked.connect(self.connect_camera_tl)
        self.pushButton_disconnect_TL.clicked.connect(self.disconnect_camera_tl)
        self.pushButton_LiveView_TL.clicked.connect(self.toggle_live_view_tl)
        self.pushButton_snap_TL.clicked.connect(self.snap_image_tl)
        self.pushButton_createROI_TL.clicked.connect(self.toggle_roi_mode_tl)
        self.camera_TL = None
        self.live_thread_TL = None
        self.roi_enabled_TL = False
        self.roi_rect_TL = None
        self._image_initialized_TL = False
        # Connect spinbox signals once
        self.spinBox_hstart_TL.editingFinished.connect(self._update_roi_from_spinboxes_tl)
        self.spinBox_hend_TL.editingFinished.connect(self._update_roi_from_spinboxes_tl)
        self.spinBox_vstart_TL.editingFinished.connect(self._update_roi_from_spinboxes_tl)
        self.spinBox_vend_TL.editingFinished.connect(self._update_roi_from_spinboxes_tl)

        # =====================================================================
        # Filter Wheel
        # =====================================================================
        self.filter_wheel = FilterWheel(HARDWARE_CONFIG['filter_wheel_serial'])
        self.pushButton_connect_FilterWheel.clicked.connect(self.connect_filter_wheel)
        self.filter_wheel_connected = False
        # Default filter selections: 488nm -> ET 525/50 (pos 3), 561nm -> ET 605/52 (pos 2)
        self.comboBox_filterWheel_488.setCurrentIndex(3)
        self.comboBox_filterWheel_561.setCurrentIndex(2)
        self.pushButton_move_filterWheel.clicked.connect(self.move_filter_wheel_manual)
        self.pushButton_move_filterWheel.setEnabled(False)

        # =====================================================================
        # Flipper Mirrors (Thorlabs MFF101)
        # =====================================================================
        self.flipper_illumination = MFF101(HARDWARE_CONFIG['flipper_illumination_serial'])
        self.flipper_detection = MFF101(HARDWARE_CONFIG['flipper_detection_serial'])
        self.flippers_connected = False

        self.pushButton_connect_Flippers.clicked.connect(self.connect_flippers)

        # Illumination: widefield = pos 1, lightsheet = pos 2
        self.radioButton_illumination_lightsheet.toggled.connect(
            lambda checked: self._set_flipper_illumination(2) if checked else None
        )
        self.radioButton_illumination_wf.toggled.connect(
            lambda checked: self._set_flipper_illumination(1) if checked else None
        )

        # Detection: wfPCO = pos 1, PCO-only = pos 2
        self.radioButton_detection_wfPCO.toggled.connect(
            lambda checked: self._set_flipper_detection(1) if checked else None
        )
        self.radioButton_detection_PCO.toggled.connect(
            lambda checked: self._set_flipper_detection(2) if checked else None
        )

        # Disable radio buttons until flippers are connected
        self.radioButton_illumination_lightsheet.setEnabled(False)
        self.radioButton_illumination_wf.setEnabled(False)
        self.radioButton_detection_wfPCO.setEnabled(False)
        self.radioButton_detection_PCO.setEnabled(False)

        # =====================================================================
        # Scanning
        # =====================================================================
        self.pushButton_GS_run.clicked.connect(self.start_scan_multichannel)
        self.pushButton_stop.clicked.connect(self.stop_scan_multichannel)
        self.pushButton_browse_488.clicked.connect(self.browse_output_directory)
        self.pushButton_browse_561.clicked.connect(self.browse_output_directory)
        self.ao = None
        self.ao_connected = False
        self.scan_thread = None
        self.scan_worker = None

        # =====================================================================
        # Pixel info on mouse hover
        # =====================================================================
        self._current_image = None
        self._mouse_proxy = SignalProxy(
            self.widget_ImageView.scene.sigMouseMoved,
            rateLimit=30,
            slot=self._on_mouse_moved,
        )

    # =========================================================================
    # Close event — resource cleanup
    # =========================================================================
    def closeEvent(self, event):
        print("[GUI] Closing resources...")

        if self.live_thread:
            self.stop_live_view_pco()
        if self.camera is not None:
            try:
                self.camera.close()
                print("[GUI] PCO camera closed.")
            except Exception as e:
                print(f"[GUI] Error closing PCO camera: {e}")

        if self.live_thread_TL:
            self.stop_live_view_tl()
        if self.camera_TL is not None:
            try:
                self.camera_TL.close()
                print("[GUI] Thorlabs camera closed.")
            except Exception as e:
                print(f"[GUI] Error closing Thorlabs: {e}")

        if self.ao is not None:
            try:
                self.ao.close()
                print("[GUI] DAQ closed.")
            except Exception as e:
                print(f"[GUI] Error closing DAQ: {e}")

        if self.laser488_connected:
            try:
                self.laser488.finalize()
                print("[GUI] Laser 488 closed.")
            except Exception as e:
                print(f"[GUI] Error closing laser 488: {e}")

        if self.laser561_connected:
            try:
                self.laser561.finalize()
                print("[GUI] Laser 561 closed.")
            except Exception as e:
                print(f"[GUI] Error closing laser 561: {e}")

        if self.stage_connected:
            try:
                self.stage.finalize()
                print("[GUI] Stage closed.")
            except Exception as e:
                print(f"[GUI] Error closing stage: {e}")

        if self.filter_wheel_connected:
            try:
                self.filter_wheel.finalize()
                print("[GUI] Filter wheel closed.")
            except Exception as e:
                print(f"[GUI] Error closing filter wheel: {e}")

        if self.flippers_connected:
            for name, flipper in [("illumination", self.flipper_illumination),
                                  ("detection",    self.flipper_detection)]:
                try:
                    flipper.finalize()
                    print(f"[GUI] Flipper {name} closed.")
                except Exception as e:
                    print(f"[GUI] Error closing flipper {name}: {e}")

        event.accept()

    # =========================================================================
    # Laser 488
    # =========================================================================
    def toggle_laser_488(self):
        if self.laser488_connected:
            try:
                self.laser488.finalize()
                print("[GUI] Laser 488 disconnected.")
            except Exception as e:
                print(f"[GUI] Error disconnecting laser 488: {e}")
            self.laser488_connected = False
            self.tab488.setEnabled(False)
            self.pushButton_connect_488.setText("Connect 488")
        else:
            self.pushButton_connect_488.setEnabled(False)
            self.thread_488 = SerialDeviceInitThread(self.laser488)
            self.thread_488.finished.connect(self._on_laser_488_connected)
            self.thread_488.start()

    def _on_laser_488_connected(self, serial_number):
        print(f'The 488 device serial number is: {serial_number}')
        self.tab488.setEnabled(True)
        self.pushButton_connect_488.setEnabled(True)
        self.pushButton_connect_488.setText("Disconnect 488")
        self.laser488_connected = True
        self.refresh_488()

    def command_488(self):
        cmd_message = self.lineEdit_cmd_488.text()
        reply = self.laser488.query(cmd_message)
        self.label_asw_488.setText(reply)

    def set_power_488(self):
        power = self.spinBox_power_488.value()
        result = self.laser488.set_laser_power(power)
        print(f"[Laser 488] Set power: {result}")

    def set_current_488(self):
        current = self.spinBox_current_488.value()
        result = self.laser488.set_diode_current(current)
        print(f"[Laser 488] Set current: {result}")

    def refresh_488(self):
        try:
            self.spinBox_power_488.setValue(float(self.laser488.ask_power()))
            self.spinBox_current_488.setValue(float(self.laser488.ask_current()))
            self.label_tempBase_488.setText(str(self.laser488.ask_base_temp()))
            self.label_tempDiode_488.setText(str(self.laser488.ask_diode_temp()))
            self.lcdNumber_488.display(int(self.laser488.ask_hours()))

            control_mode = self.laser488.ask_analog_control_mode()
            if control_mode == "Power":
                self.radioButton_cteP_488.setChecked(True)
                self.spinBox_current_488.setEnabled(False)
                self.pushButton_setC_488.setEnabled(False)
            elif control_mode == "Current":
                self.radioButton_cteC_488.setChecked(True)
                self.spinBox_power_488.setEnabled(False)
                self.pushButton_setP_488.setEnabled(False)

            self.label_status_488.setText(self.laser488.ask_status())
        except Exception as e:
            print(f"[Laser 488] Error refreshing: {e}")

    def set_constant_power_mode_488(self):
        if self.laser488_connected:
            self.laser488.set_laser_mode("Power")
            self.spinBox_power_488.setEnabled(True)
            self.pushButton_setP_488.setEnabled(True)
            self.spinBox_current_488.setEnabled(False)
            self.pushButton_setC_488.setEnabled(False)

    def set_constant_current_mode_488(self):
        if self.laser488_connected:
            self.laser488.set_laser_mode("Current")
            self.spinBox_current_488.setEnabled(True)
            self.pushButton_setC_488.setEnabled(True)
            self.spinBox_power_488.setEnabled(False)
            self.pushButton_setP_488.setEnabled(False)

    # =========================================================================
    # Laser 561
    # =========================================================================
    def toggle_laser_561(self):
        if self.laser561_connected:
            try:
                self.laser561.finalize()
                print("[GUI] Laser 561 disconnected.")
            except Exception as e:
                print(f"[GUI] Error disconnecting laser 561: {e}")
            self.laser561_connected = False
            self.tab561.setEnabled(False)
            self.pushButton_connect_561.setText("Connect 561")
        else:
            self.pushButton_connect_561.setEnabled(False)
            self.thread_561 = SerialDeviceInitThread(self.laser561)
            self.thread_561.finished.connect(self._on_laser_561_connected)
            self.thread_561.start()

    def _on_laser_561_connected(self, serial_number):
        print(f'The 561 device serial number is: {serial_number}')
        self.tab561.setEnabled(True)
        self.pushButton_connect_561.setEnabled(True)
        self.pushButton_connect_561.setText("Disconnect 561")
        self.laser561_connected = True
        self.refresh_561()

    def _confirm_laser_561_off(self):
        reply = QMessageBox.question(
            self, "Turn off laser 561",
            "Are you sure you want to turn off the 561 laser?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.laser561.set_laser_control("OFF")

    def command_561(self):
        cmd_message = self.lineEdit_cmd_561.text()
        reply = self.laser561.query(cmd_message)
        self.label_asw_561.setText(reply)

    def refresh_561(self):
        try:
            self.spinBox_power_561.setValue(float(self.laser561.ask_power()))
        except Exception as e:
            print(f"[Laser 561] Error refreshing: {e}")

    # =========================================================================
    # Stage
    # =========================================================================
    def connect_stage(self):
        self.pushButton_connect_Stage.setEnabled(False)
        self.thread_stage = SerialDeviceInitThread(self.stage)
        self.thread_stage.finished.connect(self._on_stage_connected)
        self.thread_stage.start()

    def _on_stage_connected(self, serial_number):
        print(f'The stage device serial number is: {serial_number}')
        self.groupBox_moves_Stage.setEnabled(True)
        self.pushButton_connect_Stage.setEnabled(True)
        self.stage_connected = True
        self.refresh_stage()

    def refresh_stage(self):
        try:
            self.label_statusX.setText(f"{self.stage.ask_position_um('X') / 1000:.4f}")
            self.label_statusY.setText(f"{self.stage.ask_position_um('Y') / 1000:.4f}")
            self.label_statusZ.setText(f"{self.stage.ask_position_um('Z') / 1000:.4f}")
        except Exception as e:
            print(f"[Stage] Error refreshing: {e}")

    def move_relative_xprime(self, direction):
        """Move in the rotated coordinate system (X')."""
        angle_rad = np.deg2rad(self.doubleSpinBox_angle.value())
        dxprime_um = self.doubleSpinBox_Xprima.value() * direction
        # Negative sign: looking at the objective, negative X is to the right
        dx = dxprime_um * np.cos(angle_rad) * (-1)
        dz = dxprime_um * np.sin(angle_rad)
        print(f'Moving X by {dx:.2f} um and Z by {dz:.2f} um')
        self.stage.move_relative_axis("X", int(dx * 10))
        self.stage.move_relative_axis("Z", int(dz * 10))
        self.refresh_stage()

    def move_relative_zprime(self, direction):
        """Move in the rotated coordinate system (Z')."""
        angle_rad = np.deg2rad(self.doubleSpinBox_angle.value())
        dzprime_um = self.doubleSpinBox_Zprima.value() * direction
        dx = dzprime_um * np.sin(angle_rad)
        dz = dzprime_um * np.cos(angle_rad)
        print(f'Moving X by {dx:.2f} um and Z by {dz:.2f} um')
        self.stage.move_relative_axis("X", int(dx * 10))
        self.stage.move_relative_axis("Z", int(dz * 10))
        self.refresh_stage()

    # =========================================================================
    # Filter Wheel
    # =========================================================================
    def connect_filter_wheel(self):
        """Start filter wheel connection in a background thread."""
        self.pushButton_connect_FilterWheel.setEnabled(False)
        self.pushButton_connect_FilterWheel.setText("Connecting...")
        self.fw_thread = FilterWheelInitThread(self.filter_wheel)
        self.fw_thread.finished.connect(self._on_filter_wheel_connected)
        self.fw_thread.start()

    def _on_filter_wheel_connected(self, success):
        """Callback when the filter wheel finishes connecting."""
        if success:
            self.filter_wheel_connected = True
            self.pushButton_connect_FilterWheel.setText("Connected")
            self.pushButton_move_filterWheel.setEnabled(True)
            pos = self.filter_wheel.get_position()
            print(f"[FilterWheel] Connected. Position: {pos} - {self.filter_wheel.get_current_filter()}")
        else:
            self.pushButton_connect_FilterWheel.setText("Connect FW")
            self.pushButton_connect_FilterWheel.setEnabled(True)
            QMessageBox.warning(self, "Error", "Could not connect the filter wheel.")

    def move_filter_wheel_to_position(self, position: int) -> bool:
        """Move the filter wheel to the specified position (1-6)."""
        if not self.filter_wheel_connected:
            print("[FilterWheel] Not connected")
            return False
        try:
            return self.filter_wheel.move_to_position(position)
        except Exception as e:
            print(f"[FilterWheel] Error moving: {e}")
            return False

    def get_filter_position_from_combobox(self, channel: str) -> int:
        """Get the filter position selected in the comboBox for a channel."""
        if channel == '488':
            return self.comboBox_filterWheel_488.currentIndex()
        elif channel == '561':
            return self.comboBox_filterWheel_561.currentIndex()
        elif channel == '647':
            return self.comboBox_filterWheel_647.currentIndex()
        return 1

    def move_filter_wheel_manual(self):
        """Move the filter wheel to the position selected in the manual comboBox."""
        if not self.filter_wheel_connected:
            QMessageBox.warning(self, "Error", "Filter wheel is not connected.")
            return

        position = self.comboBox_filterWheel.currentIndex()
        self.pushButton_move_filterWheel.setEnabled(False)
        self.pushButton_move_filterWheel.setText("Moving...")
        try:
            success = self.filter_wheel.move_to_position(position)
            if success:
                name = self.filter_wheel.get_current_filter()
                print(f"[FilterWheel] Moved to position {position} - {name}")
            else:
                QMessageBox.warning(self, "Error", f"Could not move to position {position}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error moving filter wheel: {e}")
        finally:
            self.pushButton_move_filterWheel.setEnabled(True)
            self.pushButton_move_filterWheel.setText("Move")

    # =========================================================================
    # Flipper Mirrors (MFF101)
    # =========================================================================
    def connect_flippers(self):
        """Connect both MFF101 flippers."""
        self.pushButton_connect_Flippers.setEnabled(False)
        self.pushButton_connect_Flippers.setText("Connecting...")
        errors = []
        for name, flipper in [("illumination", self.flipper_illumination),
                               ("detection",    self.flipper_detection)]:
            try:
                flipper.connect()
                print(f"[Flipper] {name} connected (SN {flipper.serial_number}).")
            except Exception as e:
                errors.append(f"{name}: {e}")
                print(f"[Flipper] Error connecting {name}: {e}")

        if errors:
            self.pushButton_connect_Flippers.setText("Connect Flippers")
            self.pushButton_connect_Flippers.setEnabled(True)
            QMessageBox.warning(self, "Flipper Error",
                                "Could not connect:\n" + "\n".join(errors))
        else:
            self.flippers_connected = True
            self.pushButton_connect_Flippers.setText("Connected")
            self.radioButton_illumination_lightsheet.setEnabled(True)
            self.radioButton_illumination_wf.setEnabled(True)
            self.radioButton_detection_wfPCO.setEnabled(True)
            self.radioButton_detection_PCO.setEnabled(True)

            # Leer posición actual y marcar el radio button sin disparar movimiento
            try:
                ill_pos = self.flipper_illumination.get_position()
                for rb in (self.radioButton_illumination_lightsheet,
                           self.radioButton_illumination_wf):
                    rb.blockSignals(True)
                if ill_pos == 2:
                    self.radioButton_illumination_lightsheet.setChecked(True)
                else:
                    self.radioButton_illumination_wf.setChecked(True)
                for rb in (self.radioButton_illumination_lightsheet,
                           self.radioButton_illumination_wf):
                    rb.blockSignals(False)
                print(f"[Flipper] Illumination posición actual: {ill_pos}")
            except Exception as e:
                print(f"[Flipper] No se pudo leer posición de iluminación: {e}")

            try:
                det_pos = self.flipper_detection.get_position()
                for rb in (self.radioButton_detection_wfPCO,
                           self.radioButton_detection_PCO):
                    rb.blockSignals(True)
                if det_pos == 1:
                    self.radioButton_detection_wfPCO.setChecked(True)
                else:
                    self.radioButton_detection_PCO.setChecked(True)
                for rb in (self.radioButton_detection_wfPCO,
                           self.radioButton_detection_PCO):
                    rb.blockSignals(False)
                print(f"[Flipper] Detection posición actual: {det_pos}")
            except Exception as e:
                print(f"[Flipper] No se pudo leer posición de detección: {e}")

    def _set_flipper_illumination(self, position: int):
        """Move the illumination flipper to position 1 (lightsheet) or 2 (WF)."""
        if not self.flippers_connected:
            return
        try:
            self.flipper_illumination.set_position(position)
            label = "lightsheet" if position == 1 else "widefield"
            print(f"[Flipper] Illumination → {label} (pos {position})")
        except Exception as e:
            QMessageBox.critical(self, "Flipper Error",
                                 f"Error moving illumination flipper: {e}")

    def _set_flipper_detection(self, position: int):
        """Move the detection flipper to position 1 (wfPCO) or 2 (PCO-only)."""
        if not self.flippers_connected:
            return
        try:
            self.flipper_detection.set_position(position)
            label = "wfPCO" if position == 1 else "PCO-only"
            print(f"[Flipper] Detection → {label} (pos {position})")
        except Exception as e:
            QMessageBox.critical(self, "Flipper Error",
                                 f"Error moving detection flipper: {e}")

    # =========================================================================
    # Thorlabs Camera
    # =========================================================================
    def connect_camera_tl(self):
        try:
            self.camera_TL = ThorlabsTLCamera()
            print(self.camera_TL.get_device_info())

            exposure_s = self.camera_TL.get_exposure()
            gain = self.camera_TL.get_gain()
            gain_min, gain_max = self.camera_TL.get_gain_range()
            frame_period_s = self.camera_TL.get_frame_period()

            self.spinBox_expTime_TL.setValue(int(exposure_s * 1000))
            self.spinBox_gain_TL.setValue(int(gain))
            self.spinBox_gain_TL.setRange(int(gain_min), int(gain_max))
            self.spinBox_frameRate_TL.setValue(int(1 / frame_period_s))

            hlim, vlim = self.camera_TL.get_roi_limits()
            self.spinBox_hstart_TL.setMaximum(hlim[1])
            self.spinBox_hend_TL.setMaximum(hlim[1])
            self.spinBox_vstart_TL.setMaximum(vlim[1])
            self.spinBox_vend_TL.setMaximum(vlim[1])
            self.spinBox_hbin_TL.setMaximum(hlim[4])
            self.spinBox_vbin_TL.setMaximum(vlim[4])

            hstart, hend, vstart, vend, hbin, vbin = self.camera_TL.get_roi()
            self.spinBox_hstart_TL.setValue(hstart)
            self.spinBox_hend_TL.setValue(hend)
            self.spinBox_vstart_TL.setValue(vstart)
            self.spinBox_vend_TL.setValue(vend)
            self.spinBox_hbin_TL.setValue(hbin)
            self.spinBox_vbin_TL.setValue(vbin)

            self.pushButton_connect_TL.setEnabled(False)
            self.pushButton_disconnect_TL.setEnabled(True)
            QMessageBox.information(self, "Thorlabs Zelux", "Camera connected successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Connection error", str(e))

    def disconnect_camera_tl(self):
        if self.camera_TL:
            try:
                if self.live_thread_TL:
                    self.stop_live_view_tl()
                self.camera_TL.close()
                self.camera_TL = None
                QMessageBox.information(self, "Disconnected", "Thorlabs camera disconnected.")
                self.pushButton_connect_TL.setEnabled(True)
                self.pushButton_disconnect_TL.setEnabled(False)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def start_live_view_tl(self):
        try:
            exp_time_s = self.spinBox_expTime_TL.value() / 1000
            frame_rate = self.spinBox_frameRate_TL.value()
            frame_period_s = 1 / frame_rate
            gain = self.spinBox_gain_TL.value()

            if frame_period_s < exp_time_s:
                QMessageBox.warning(self, "Error",
                    f"Frame period ({frame_period_s:.4f}s) must be >= exposure ({exp_time_s:.4f}s).")
                return

            self.camera_TL.set_exposure(exp_time_s)
            self.camera_TL.set_frame_period(frame_period_s)
            self.camera_TL.set_gain(gain)

            if self.roi_enabled_TL and self.roi_rect_TL:
                hstart = self.spinBox_hstart_TL.value()
                hend = self.spinBox_hend_TL.value()
                vstart = self.spinBox_vstart_TL.value()
                vend = self.spinBox_vend_TL.value()
                if hend <= hstart or vend <= vstart:
                    QMessageBox.warning(self, "Invalid ROI", "ROI values are not valid.")
                    return
                self.camera_TL.set_roi(hstart=hstart, hend=hend, vstart=vstart, vend=vend)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not start live view:\n{e}")
            return

        self.live_thread_TL = LiveViewThread(self.camera_TL)
        self.live_thread_TL.image_ready.connect(self.update_image_tl)
        self.live_thread_TL.start()

    def stop_live_view_tl(self):
        if self.live_thread_TL:
            self.live_thread_TL.stop()
            self.live_thread_TL = None

    def toggle_live_view_tl(self):
        if self.live_thread_TL and self.live_thread_TL.isRunning():
            self.stop_live_view_tl()
            self.pushButton_LiveView_TL.setText("Start Live View")
        else:
            self.start_live_view_tl()
            self.pushButton_LiveView_TL.setText("Stop Live View")

    def update_image_tl(self, img):
        if img is None:
            return
        img = img.T
        self._current_image = img
        if not self._image_initialized_TL:
            self.widget_ImageView.setImage(img, autoLevels=False, levels=(0, 255))
            self._image_initialized_TL = True
        else:
            self.widget_ImageView.imageItem.setImage(img, autoLevels=False, levels=(0, 255))

        if self.checkBox_roi_intensity_TL.isChecked() and self.roi_enabled_TL and self.roi_rect_TL:
            self._show_roi_intensity(img, self.roi_rect_TL, self.label_roi_intensity_TL)

    def snap_image_tl(self):
        try:
            exp_time_s = self.spinBox_expTime_TL.value() / 1000
            gain = self.spinBox_gain_TL.value()
            self.camera_TL.set_exposure(exp_time_s)
            self.camera_TL.set_gain(gain)

            if self.roi_enabled_TL and self.roi_rect_TL:
                hstart = self.spinBox_hstart_TL.value()
                hend = self.spinBox_hend_TL.value()
                vstart = self.spinBox_vstart_TL.value()
                vend = self.spinBox_vend_TL.value()
                if hend <= hstart or vend <= vstart:
                    QMessageBox.warning(self, "Invalid ROI", "ROI values are not valid.")
                    return
                self.camera_TL.set_roi(hstart=hstart, hend=hend, vstart=vstart, vend=vend)

            img = self.camera_TL.snap(timeout=5.0)
            if img is not None:
                self.update_image_tl(img)
            else:
                QMessageBox.warning(self, "Snap failed", "Could not acquire image.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Snap failed:\n{e}")

    def toggle_roi_mode_tl(self):
        if self.roi_enabled_TL:
            if self.roi_rect_TL is not None:
                self.widget_ImageView.removeItem(self.roi_rect_TL)
                self.roi_rect_TL = None
            self.roi_enabled_TL = False
            self.pushButton_createROI_TL.setText("Create ROI")
        else:
            self.roi_rect_TL = RectROI([300, 300], [700, 700], movable=True, resizable=True)
            self.widget_ImageView.addItem(self.roi_rect_TL)
            self.roi_enabled_TL = True
            self.pushButton_createROI_TL.setText("Cancel ROI")
            self.roi_rect_TL.sigRegionChanged.connect(self._update_spinboxes_from_roi_tl)
            self._update_spinboxes_from_roi_tl()

    def _update_spinboxes_from_roi_tl(self):
        if not self.roi_rect_TL:
            return
        pos = self.roi_rect_TL.pos()
        size = self.roi_rect_TL.size()
        for sb in (self.spinBox_hstart_TL, self.spinBox_hend_TL,
                    self.spinBox_vstart_TL, self.spinBox_vend_TL):
            sb.blockSignals(True)
        self.spinBox_hstart_TL.setValue(int(pos.x()))
        self.spinBox_hend_TL.setValue(int(pos.x() + size.x()))
        self.spinBox_vstart_TL.setValue(int(pos.y()))
        self.spinBox_vend_TL.setValue(int(pos.y() + size.y()))
        for sb in (self.spinBox_hstart_TL, self.spinBox_hend_TL,
                    self.spinBox_vstart_TL, self.spinBox_vend_TL):
            sb.blockSignals(False)

    def _update_roi_from_spinboxes_tl(self):
        if not self.roi_enabled_TL or not self.roi_rect_TL:
            return
        x1, x2 = self.spinBox_hstart_TL.value(), self.spinBox_hend_TL.value()
        y1, y2 = self.spinBox_vstart_TL.value(), self.spinBox_vend_TL.value()
        self.roi_rect_TL.setPos([min(x1, x2), min(y1, y2)])
        self.roi_rect_TL.setSize([abs(x2 - x1), abs(y2 - y1)])

    # =========================================================================
    # PCO Camera
    # =========================================================================
    def connect_camera_pco(self):
        try:
            self.camera = camera_pco.Camera()

            # Populate GUI from camera state
            self.spinBox_exptime_PCO.setValue(int(self.camera.exposure_us / 1000))
            frame_period_us = self.camera.exposure_us + self.camera.rolling_time_us
            if frame_period_us > 0:
                self.spinBox_framerate_PCO.setValue(int(1e6 / frame_period_us))

            # Crop ROI spinbox limits (full sensor)
            self.spinBox_hstart.setMaximum(self.camera.width_px)
            self.spinBox_hend.setMaximum(self.camera.width_px)
            self.spinBox_vstart.setMaximum(self.camera.height_px)
            self.spinBox_vend.setMaximum(self.camera.height_px)
            self.spinBox_hbin.setMaximum(1)
            self.spinBox_vbin.setMaximum(1)

            # Default crop: full image (no crop)
            self.spinBox_hstart.setValue(0)
            self.spinBox_hend.setValue(self.camera.width_px)
            self.spinBox_vstart.setValue(0)
            self.spinBox_vend.setValue(self.camera.height_px)
            self.spinBox_hbin.setValue(1)
            self.spinBox_vbin.setValue(1)

            self.camera_connected = True
            self.pushButton_connect_PCO.setEnabled(False)
            self.pushButton_disconnect_PCO.setEnabled(True)
            QMessageBox.information(self, "PCO Edge", "Camera connected successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Connection error", str(e))

    def disconnect_camera_pco(self):
        if self.camera:
            try:
                if self.live_thread:
                    self.stop_live_view_pco()
                self.camera.close()
                self.camera = None
                self.camera_connected = False
                self._image_initialized = False
                QMessageBox.information(self, "Disconnected", "PCO camera disconnected.")
                self.pushButton_connect_PCO.setEnabled(True)
                self.pushButton_disconnect_PCO.setEnabled(False)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def start_live_view_pco(self):
        if self.camera is None:
            QMessageBox.warning(self, "Error", "Camera not connected.")
            return
        try:
            exp_time_s = self.spinBox_exptime_PCO.value() / 1000
            self.camera.set_exposure(exp_time_s)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not start live view:\n{e}")
            return

        self.live_thread = LiveViewThread(self.camera)
        self.live_thread.image_ready.connect(self.update_image_pco)
        self.live_thread.start()

    def stop_live_view_pco(self):
        if self.live_thread:
            self.live_thread.stop()
            self.live_thread = None

    def toggle_live_view_pco(self):
        if self.live_thread and self.live_thread.isRunning():
            self.stop_live_view_pco()
            self.pushButton_LiveView_PCO.setText("Start Live View")
        else:
            self.start_live_view_pco()
            self.pushButton_LiveView_PCO.setText("Stop Live View")

    def update_image_pco(self, img):
        if img is None:
            return
        img = img.T
        self._current_image = img
        if not self._image_initialized:
            self.widget_ImageView.setImage(img, autoLevels=False, levels=(0, 65535))
            self._image_initialized = True
        else:
            self.widget_ImageView.imageItem.setImage(img, autoLevels=False, levels=(0, 65535))

        if self.checkBox_roi_intensity_PCO.isChecked() and self.roi_enabled and self.roi_rect:
            self._show_roi_intensity(img, self.roi_rect, self.label_roi_intensity_PCO)
        if self.checkBox_roi_max_intensity_PCO.isChecked() and self.roi_enabled and self.roi_rect:
            self._show_roi_max_intensity(img, self.roi_rect, self.label_roi_max_intensity_PCO)
        if self.checkBox_roi_sum_intensity_PCO.isChecked() and self.roi_enabled and self.roi_rect:
            self._show_roi_sum_intensity(img, self.roi_rect, self.label_roi_sum_intensity_PCO)

    def snap_image_pco(self):
        if self.camera is None:
            QMessageBox.warning(self, "Error", "Camera not connected.")
            return
        try:
            exposure_us = self.spinBox_exptime_PCO.value() * 1000
            img = self.camera.snap(exposure_us=exposure_us)
            if img is not None:
                self.update_image_pco(img)
            else:
                QMessageBox.warning(self, "Snap failed", "Could not acquire image.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Snap failed:\n{e}")

    def toggle_roi_mode_pco(self):
        if self.roi_enabled:
            if self.roi_rect is not None:
                self.widget_ImageView.removeItem(self.roi_rect)
                self.roi_rect = None
            self.roi_enabled = False
            self.pushButton_createROI_PCO.setText("Create ROI")
        else:
            self.roi_rect = RectROI([300, 300], [700, 700], movable=True, resizable=True)
            self.widget_ImageView.addItem(self.roi_rect)
            self.roi_enabled = True
            self.pushButton_createROI_PCO.setText("Cancel ROI")
            self.roi_rect.sigRegionChanged.connect(self._update_spinboxes_from_roi_pco)
            self._update_spinboxes_from_roi_pco()

    def _update_spinboxes_from_roi_pco(self):
        if not self.roi_rect:
            return
        pos = self.roi_rect.pos()
        size = self.roi_rect.size()
        for sb in (self.spinBox_hstart, self.spinBox_hend,
                    self.spinBox_vstart, self.spinBox_vend):
            sb.blockSignals(True)
        self.spinBox_hstart.setValue(int(pos.x()))
        self.spinBox_hend.setValue(int(pos.x() + size.x()))
        self.spinBox_vstart.setValue(int(pos.y()))
        self.spinBox_vend.setValue(int(pos.y() + size.y()))
        for sb in (self.spinBox_hstart, self.spinBox_hend,
                    self.spinBox_vstart, self.spinBox_vend):
            sb.blockSignals(False)

    def _update_roi_from_spinboxes_pco(self):
        if not self.roi_enabled or not self.roi_rect:
            return
        x1, x2 = self.spinBox_hstart.value(), self.spinBox_hend.value()
        y1, y2 = self.spinBox_vstart.value(), self.spinBox_vend.value()
        self.roi_rect.setPos([min(x1, x2), min(y1, y2)])
        self.roi_rect.setSize([abs(x2 - x1), abs(y2 - y1)])

    # =========================================================================
    # Shared ROI helper
    # =========================================================================
    @staticmethod
    def _show_roi_intensity(img, roi_rect, label):
        """Compute and display mean intensity inside a ROI.
        img is already transposed (shape = W x H), so first axis = x, second = y."""
        try:
            pos = roi_rect.pos()
            size = roi_rect.size()
            x1 = max(0, int(pos.x()))
            x2 = min(img.shape[0], int(pos.x() + size.x()))
            y1 = max(0, int(pos.y()))
            y2 = min(img.shape[1], int(pos.y() + size.y()))
            roi_img = img[x1:x2, y1:y2]
            label.setText(f"{roi_img.mean():.2f}")
        except Exception as e:
            print(f"Error computing ROI intensity: {e}")
            label.setText("Err")

    @staticmethod
    def _show_roi_max_intensity(img, roi_rect, label):
        """Compute and display max intensity inside a ROI.
        img is already transposed (shape = W x H), so first axis = x, second = y."""
        try:
            pos = roi_rect.pos()
            size = roi_rect.size()
            x1 = max(0, int(pos.x()))
            x2 = min(img.shape[0], int(pos.x() + size.x()))
            y1 = max(0, int(pos.y()))
            y2 = min(img.shape[1], int(pos.y() + size.y()))
            roi_img = img[x1:x2, y1:y2]
            label.setText(f"{roi_img.max():.0f}")
        except Exception as e:
            print(f"Error computing ROI max intensity: {e}")
            label.setText("Err")

    @staticmethod
    def _show_roi_sum_intensity(img, roi_rect, label):
        """Compute and display sum intensity inside a ROI.
        img is already transposed (shape = W x H), so first axis = x, second = y."""
        try:
            pos = roi_rect.pos()
            size = roi_rect.size()
            x1 = max(0, int(pos.x()))
            x2 = min(img.shape[0], int(pos.x() + size.x()))
            y1 = max(0, int(pos.y()))
            y2 = min(img.shape[1], int(pos.y() + size.y()))
            roi_img = img[x1:x2, y1:y2]
            label.setText(f"{int(roi_img.sum())}")
        except Exception as e:
            print(f"Error computing ROI sum intensity: {e}")
            label.setText("Err")

    # =========================================================================
    # Pixel info on mouse hover
    # =========================================================================
    def _on_mouse_moved(self, event):
        pos = event[0]
        view = self.widget_ImageView.getView()
        img = self._current_image
        if img is None or not view.sceneBoundingRect().contains(pos):
            self.statusBar().clearMessage()
            return
        mouse_point = view.mapSceneToView(pos)
        x = int(mouse_point.x())
        y = int(mouse_point.y())
        if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
            self.statusBar().showMessage(f"X: {x}   Y: {y}   Val: {img[x, y]}")
        else:
            self.statusBar().clearMessage()

    # =========================================================================
    # Scanning
    # =========================================================================
    def browse_output_directory(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select output directory", "",
            QFileDialog.DontUseNativeDialog,
        )
        if folder:
            self.lineEdit_directory_488.setText(folder)
            self.lineEdit_directory_561.setText(folder)

    def start_scan_multichannel(self):
        """Start multichannel acquisition with the temporal scheduler."""
        output_dir_488 = self.lineEdit_directory_488.text().strip()
        output_dir_561 = self.lineEdit_directory_561.text().strip()

        if self.checkBox_on_488.isChecked() and not output_dir_488:
            QMessageBox.warning(self, "Missing directory",
                "Channel 488 is enabled but no output directory was specified.")
            return
        if self.checkBox_on_561.isChecked() and not output_dir_561:
            QMessageBox.warning(self, "Missing directory",
                "Channel 561 is enabled but no output directory was specified.")
            return

        for output_dir in [output_dir_488, output_dir_561]:
            if output_dir and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                    print(f"[INFO] Directory created: {output_dir}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Could not create directory:\n{output_dir}\n{e}")
                    return

        if self.live_thread:
            print("[INFO] Stopping LiveView before starting scan...")
            self.stop_live_view_pco()

        if self.camera is None:
            try:
                self.camera = camera_pco.Camera()
                self.camera_connected = True
                print("[INFO] PCO camera connected for scanning")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not connect camera:\n{e}")
                return

        if self.ao is None:
            try:
                self.ao = daq.DAQ(
                    num_channels=HARDWARE_CONFIG['daq_num_channels'],
                    rate=HARDWARE_CONFIG['daq_rate'],
                )
                self.ao_connected = True
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not connect DAQ:\n{e}")
                return

        self.pushButton_GS_run.setEnabled(False)

        roi_hstart = self.spinBox_hstart.value()
        roi_hend = self.spinBox_hend.value()
        roi_vstart = self.spinBox_vstart.value()
        roi_vend = self.spinBox_vend.value()

        config = {}
        for channel in ['561', '488']:
            if channel == '561':
                enabled = self.checkBox_on_561.isChecked()
                config[channel] = {
                    'enabled': enabled,
                    'save_raw': self.checkBox_saveRaw_561.isChecked(),
                    'save_crop': self.checkBox_saveCrop_561.isChecked(),
                    'save_deskew': self.checkBox_saveDeskew_561.isChecked(),
                    'image_path': self.lineEdit_directory_561.text(),
                    'scan_range_um': self.spinBox_range_561.value(),
                    'exposure_ms': self.spinBox_expTime_561.value(),
                    'num_volumes': self.spinBox_cant_vol_561.value(),
                    'interval_s': self.spinBox_tiempo_entre_vol_561.value(),
                    'drift_correction':            self.checkBox_driftCorrection_561.isChecked(),
                    'drift_correction_every':      self.spinBox_driftCorrection_561.value(),
                    'drift_correction_type':       self.comboBox_typeDriftCorrection_561.currentText(),
                    'drift_correction_ref_update': self.spinBox_refDriftCorrection_561.value(),
                    'roi_hstart': roi_hstart, 'roi_hend': roi_hend,
                    'roi_vstart': roi_vstart, 'roi_vend': roi_vend,
                    'filter_position': self.get_filter_position_from_combobox('561'),
                    'z_stage_enabled': self.checkBox_ZstageScan_561.isChecked(),
                    'z_stage_steps': self.spinBox_ZstageScan_561_steps.value(),
                    'z_stage_step_um': self.spinBox_ZstageScan_561_distance.value(),
                }
            elif channel == '488':
                enabled = self.checkBox_on_488.isChecked()
                config[channel] = {
                    'enabled': enabled,
                    'save_raw': self.checkBox_saveRaw_488.isChecked(),
                    'save_crop': self.checkBox_saveCrop_488.isChecked(),
                    'save_deskew': self.checkBox_saveDeskew_488.isChecked(),
                    'image_path': self.lineEdit_directory_488.text(),
                    'scan_range_um': self.spinBox_range_488.value(),
                    'exposure_ms': self.spinBox_expTime_488.value(),
                    'num_volumes': self.spinBox_cant_vol_488.value(),
                    'interval_s': self.spinBox_tiempo_entre_vol_488.value(),
                    'drift_correction':            self.checkBox_driftCorrection_488.isChecked(),
                    'drift_correction_every':      self.spinBox_driftCorrection_488.value(),
                    'drift_correction_type':       self.comboBox_typeDriftCorrection_488.currentText(),
                    'drift_correction_ref_update': self.spinBox_refDriftCorrection_488.value(),
                    'roi_hstart': roi_hstart, 'roi_hend': roi_hend,
                    'roi_vstart': roi_vstart, 'roi_vend': roi_vend,
                    'filter_position': self.get_filter_position_from_combobox('488'),
                    'z_stage_enabled': self.checkBox_ZstageScan_488.isChecked(),
                    'z_stage_steps': self.spinBox_ZstageScan_488_steps.value(),
                    'z_stage_step_um': self.spinBox_ZstageScan_488_distance.value(),
                }

        # Parámetros ópticos globales: leídos de la GUI para que sean consistentes
        # entre el movimiento oblicuo del stage y el deskew.
        config['galvo_step_um'] = self.doubleSpinBox_galvoStep.value()
        config['tilt_deg']      = self.doubleSpinBox_angle.value()

        lasers = {'561': self.laser561, '488': self.laser488}

        if config['488']['enabled'] and not self.laser488_connected:
            QMessageBox.warning(self, "Laser not connected",
                "Channel 488 is enabled but the 488 laser is not connected.")
            self.pushButton_GS_run.setEnabled(True)
            return
        if config['561']['enabled'] and not self.laser561_connected:
            QMessageBox.warning(self, "Laser not connected",
                "Channel 561 is enabled but the 561 laser is not connected.")
            self.pushButton_GS_run.setEnabled(True)
            return

        z_scan_needed = any(
            params['enabled'] and params.get('z_stage_enabled', False)
            and params.get('z_stage_steps', 1) > 1
            for params in config.values() if isinstance(params, dict)
        )
        if z_scan_needed and not self.stage_connected:
            QMessageBox.warning(self, "Stage not connected",
                "Z-stage scan is enabled but the stage is not connected.")
            self.pushButton_GS_run.setEnabled(True)
            return

        self.scan_worker = MultichannelSchedulerWorker(
            camera=self.camera,
            ao=self.ao,
            stage=self.stage,
            lasers=lasers,
            filter_wheel=self.filter_wheel if self.filter_wheel_connected else None,
            config=config,
        )
        self.scan_thread = QThread()
        self.scan_worker.moveToThread(self.scan_thread)

        self.scan_thread.started.connect(self.scan_worker.run)
        self.scan_worker.finished.connect(self._on_scan_finished)
        self.scan_worker.finished.connect(self.scan_thread.quit)
        self.scan_worker.finished.connect(self.scan_worker.deleteLater)
        self.scan_thread.finished.connect(self.scan_thread.deleteLater)
        self.scan_worker.log_message.connect(lambda msg: print(msg))

        print("[MainWindow] Starting multichannel scan with temporal scheduler")
        print(f"[MainWindow] Crop ROI: hstart={roi_hstart}, hend={roi_hend}, vstart={roi_vstart}, vend={roi_vend}")
        self.scan_thread.start()

    def stop_scan_multichannel(self):
        if self.scan_worker:
            print("[MainWindow] Stop requested")
            self.scan_worker.request_stop()

    def _on_scan_finished(self):
        print("[GUI] Scan finished.")
        self.pushButton_GS_run.setEnabled(True)
        # Camera stays open — user can start liveview without reconnecting

        if self.ao is not None:
            try:
                self.ao.close()
                self.ao = None
                self.ao_connected = False
                print("[GUI] DAQ closed.")
            except Exception as e:
                print(f"[GUI] Error closing DAQ: {e}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    m = MainWindow()
    m.show()
    sys.exit(app.exec_())
