"""
OPM Acquisition Workers
=======================

Workers de adquisición desacoplados de la GUI.

Clases exportadas:
    - LiveViewThread              — vista en vivo de la cámara
    - SerialDeviceInitThread      — inicialización de dispositivos seriales
    - FilterWheelInitThread       — inicialización de la rueda de filtros
    - MultichannelSchedulerWorker — adquisición multicanal con scheduler temporal

Todas las clases se comunican con la GUI exclusivamente mediante señales Qt
(pyqtSignal). No importan ni dependen de ningún widget.
"""

import os
import time
import datetime

import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from skimage.registration import phase_cross_correlation
from tifffile import imwrite

from config import HARDWARE_CONFIG, DEFAULT_CONFIG
from deskew_ShepherdLab import deskew, max_projection_z


# =============================================================================
# LiveViewThread
# =============================================================================

class LiveViewThread(QThread):
    """Thread para la vista en vivo de la cámara.

    Compatible con la cámara PCO y la Thorlabs.
    """
    image_ready = pyqtSignal(np.ndarray)

    def __init__(self, camera):
        super().__init__()
        self.camera   = camera
        self._running = False

    def run(self):
        self._running = True
        try:
            self.camera.setup_acquisition(nframes=100)
            self.camera.start_acquisition()
        except Exception as e:
            print(f"Error starting acquisition: {e}")
            return

        while self._running:
            try:
                img = self.camera.read_newest_image()
                if img is not None:
                    self.image_ready.emit(img)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Error getting image: {e}")
                time.sleep(0.1)

        try:
            self.camera.stop_acquisition()
        except Exception as e:
            print(f"Error stopping acquisition: {e}")

    def stop(self):
        self._running = False
        self.wait()


# =============================================================================
# SerialDeviceInitThread
# =============================================================================

class SerialDeviceInitThread(QThread):
    """Thread genérico para inicializar dispositivos seriales (láseres, escenario).

    Llama a device.initialize() luego a device.idn() y emite el resultado.
    """
    finished = pyqtSignal(str)

    def __init__(self, device):
        super().__init__()
        self.device = device

    def run(self):
        self.device.initialize()
        serial_number = self.device.idn()
        self.finished.emit(serial_number)


# =============================================================================
# FilterWheelInitThread
# =============================================================================

class FilterWheelInitThread(QThread):
    """Thread para inicializar la rueda de filtros."""
    finished = pyqtSignal(bool)

    def __init__(self, filter_wheel):
        super().__init__()
        self.filter_wheel = filter_wheel

    def run(self):
        try:
            self.filter_wheel.connect()
            self.finished.emit(True)
        except Exception as e:
            print(f"[FilterWheel] Connection failed: {e}")
            self.finished.emit(False)


# =============================================================================
# MultichannelSchedulerWorker
# =============================================================================

class MultichannelSchedulerWorker(QObject):
    """Worker de adquisición multicanal con scheduling temporal.

    Cada canal tiene su propio número de volúmenes e intervalo. Ambos canales
    empiezan en t=0. Si dos canales coinciden, se adquieren secuencialmente.

    Signals
    -------
    finished : emitida al terminar (o al ser detenida).
    error    : emitida si ocurre una excepción no recuperable.
    log_message : mensajes de estado para mostrar en la GUI.
    """
    finished    = pyqtSignal()
    error       = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def __init__(self, camera, ao, stage, lasers, filter_wheel, config):
        super().__init__()
        self._is_running = True

        self.camera       = camera
        self.ao           = ao
        self.stage        = stage
        self.lasers       = lasers
        self.filter_wheel = filter_wheel
        self.config       = config

        self.width_px          = DEFAULT_CONFIG['width_px']
        self.height_px         = DEFAULT_CONFIG['height_px']
        self.sample_px_um      = DEFAULT_CONFIG['sample_px_um']
        self.tilt              = np.deg2rad(config.get('tilt_deg',      DEFAULT_CONFIG['tilt_deg']))
        self.galvo_step_um     = config.get('galvo_step_um', DEFAULT_CONFIG['galvo_step_um'])
        self.drift_threshold   = DEFAULT_CONFIG['drift_threshold']
        self.safety_xy         = DEFAULT_CONFIG['safety_xy']
        self.safety_z          = DEFAULT_CONFIG['safety_z']

        self.reference_images        = {}
        self.drift_correction_counts = {}

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    def _save_metadata(self, channel, params, timestamp_str):
        """Guarda un .txt con los parámetros del scan en image_path."""
        image_path = params.get('image_path', '')
        if not image_path:
            return
        os.makedirs(image_path, exist_ok=True)

        # Nombre del filtro: intentar leerlo de la rueda, sino mostrar posición
        filter_pos = params.get('filter_position', '?')
        filter_name = f"pos {filter_pos}"
        if self.filter_wheel is not None:
            try:
                current = self.filter_wheel.get_position()
                if current == filter_pos:
                    filter_name = self.filter_wheel.get_current_filter()
            except Exception:
                pass

        tilt_deg    = self.config.get('tilt_deg',      DEFAULT_CONFIG['tilt_deg'])
        galvo_step  = self.config.get('galvo_step_um', DEFAULT_CONFIG['galvo_step_um'])
        pixel_size  = DEFAULT_CONFIG['sample_px_um']

        roi_h = f"{params.get('roi_hstart', '?')} – {params.get('roi_hend', '?')}"
        roi_v = f"{params.get('roi_vstart', '?')} – {params.get('roi_vend', '?')}"

        z_enabled = params.get('z_stage_enabled', False)
        z_steps   = params.get('z_stage_steps', 1)
        z_step_um = params.get('z_stage_step_um', 0)
        if z_enabled and z_steps > 1:
            z_info = f"yes  ({z_steps} steps × {z_step_um} µm)"
        else:
            z_info = "no"

        lines = [
            f"Acquisition metadata — channel {channel}",
            f"{'=' * 46}",
            f"Date/time       : {timestamp_str}",
            f"",
            f"--- Optical parameters ---",
            f"Tilt angle      : {tilt_deg} °",
            f"Galvo step      : {galvo_step} µm",
            f"Pixel size      : {pixel_size} µm",
            f"",
            f"--- Scan parameters ---",
            f"Exposure        : {params.get('exposure_ms', '?')} ms",
            f"Scan range      : {params.get('scan_range_um', '?')} µm",
            f"Num volumes     : {params.get('num_volumes', '?')}",
            f"Interval        : {params.get('interval_s', '?')} s",
            f"Filter          : {filter_name}",
            f"",
            f"--- ROI ---",
            f"H (columns)     : {roi_h} px",
            f"V (rows)        : {roi_v} px",
            f"",
            f"--- Save options ---",
            f"Save raw        : {params.get('save_raw', '?')}",
            f"Save crop       : {params.get('save_crop', '?')}",
            f"Save deskew     : {params.get('save_deskew', '?')}",
            f"",
            f"--- Other ---",
            f"Drift correction: {params.get('drift_correction', '?')}  (every {params.get('drift_correction_every', 1)} volumes)",
            f"Z-stage scan    : {z_info}",
            f"Image path      : {image_path}",
        ]

        fname = os.path.join(image_path, f"metadata_{channel}_{timestamp_str.replace(':', '-').replace(' ', '_')}.txt")
        try:
            with open(fname, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            self.log_message.emit(f"[{channel}] Metadata saved: {os.path.basename(fname)}")
        except Exception as e:
            self.log_message.emit(f"[{channel}] WARNING: could not save metadata: {e}")

    # -------------------------------------------------------------------------
    # Scheduler loop
    # -------------------------------------------------------------------------

    def run(self):
        try:
            channels_state = {}
            for channel, params in self.config.items():
                if not isinstance(params, dict):
                    continue
                if params['enabled']:
                    channels_state[channel] = {
                        'next_time':     0.0,
                        'volumes_done':  0,
                        'total_volumes': params['num_volumes'],
                        'interval':      params['interval_s'],
                        'config':        params,
                    }
                    self.reference_images[channel]        = None
                    self.drift_correction_counts[channel] = 0

            if not channels_state:
                self.log_message.emit("[Scheduler] No channels enabled")
                self.finished.emit()
                return

            self.log_message.emit("\n" + "=" * 60)
            self.log_message.emit("[Scheduler] ACQUISITION PLAN:")
            for ch, state in channels_state.items():
                cfg    = state['config']
                z_info = ""
                if cfg.get('z_stage_enabled') and cfg.get('z_stage_steps', 1) > 1:
                    z_info = (
                        f", Z-stage: {cfg['z_stage_steps']} steps x "
                        f"{cfg['z_stage_step_um']} um"
                    )
                self.log_message.emit(
                    f"  Channel {ch}: {state['total_volumes']} volumes, "
                    f"interval {state['interval']}s{z_info}"
                )
            self.log_message.emit("=" * 60 + "\n")

            # Guardar metadata de cada canal habilitado
            timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for ch, state in channels_state.items():
                self._save_metadata(ch, state['config'], timestamp_str)

            start_time = time.time()

            while self._is_running:
                current_time = time.time() - start_time

                all_done = all(
                    s['volumes_done'] >= s['total_volumes']
                    for s in channels_state.values()
                )
                if all_done:
                    self.log_message.emit("\n[Scheduler] All channels completed!")
                    break

                channels_to_acquire = [
                    ch for ch, s in channels_state.items()
                    if s['volumes_done'] < s['total_volumes']
                    and current_time >= s['next_time']
                ]

                if not channels_to_acquire:
                    time.sleep(0.05)
                    continue

                # Separate channels that need a z-stage scan from those that don't.
                # When more than one channel is due AND all of them use the z-stage,
                # acquire them interleaved (both colors at each z-position) instead
                # of sequentially (all z-positions for one color, then the other).
                zscan_due = [
                    ch for ch in channels_to_acquire
                    if channels_state[ch]['config'].get('z_stage_enabled')
                    and channels_state[ch]['config'].get('z_stage_steps', 1) > 1
                ]
                non_zscan_due = [ch for ch in channels_to_acquire if ch not in zscan_due]

                if len(zscan_due) > 1:
                    actual_time = time.time() - start_time
                    self.log_message.emit(
                        f"\n[t={actual_time:.1f}s] Interleaved Z-scan — "
                        f"channels {zscan_due}, volumes "
                        + ", ".join(
                            f"{channels_state[ch]['volumes_done']}/"
                            f"{channels_state[ch]['total_volumes'] - 1}"
                            for ch in zscan_due
                        )
                    )
                    channels_info = [
                        (ch, channels_state[ch]['config'], channels_state[ch]['volumes_done'])
                        for ch in zscan_due
                    ]
                    self._acquire_volume_interleaved_zscan(channels_info)
                    for ch in zscan_due:
                        channels_state[ch]['volumes_done'] += 1
                        channels_state[ch]['next_time']    += channels_state[ch]['interval']
                        self.log_message.emit(
                            f"[Channel {ch}] Next volume scheduled at "
                            f"t={channels_state[ch]['next_time']:.1f}s"
                        )
                else:
                    # Single z-scan channel (or none): original sequential path
                    for channel in channels_to_acquire:
                        if not self._is_running:
                            break

                        state       = channels_state[channel]
                        vol_num     = state['volumes_done']
                        actual_time = time.time() - start_time

                        self.log_message.emit(
                            f"\n[t={actual_time:.1f}s] Channel {channel} - "
                            f"Volume {vol_num}/{state['total_volumes'] - 1} "
                            f"(scheduled: t={state['next_time']:.1f}s)"
                        )

                        self._acquire_volume(channel, state['config'], vol_num)

                        state['volumes_done'] += 1
                        state['next_time']    += state['interval']

                        self.log_message.emit(
                            f"[Channel {channel}] Next volume scheduled at "
                            f"t={state['next_time']:.1f}s"
                        )

            self.finished.emit()

        except Exception as e:
            self.log_message.emit(f"[Scheduler] ERROR: {e}")
            self.error.emit(str(e))
            self.finished.emit()

    # -------------------------------------------------------------------------
    # Interleaved multi-channel Z-stage scan
    # -------------------------------------------------------------------------

    def _acquire_volume_interleaved_zscan(self, channels_info):
        """Acquire multiple channels interleaved at each Z-stage position.

        Instead of doing all z-steps for channel A then all z-steps for channel B,
        this method moves the stage once per z-step and acquires all channels at
        that position before moving on.

        Parameters
        ----------
        channels_info : list of (channel, config, vol_index)
            Each tuple describes one channel to acquire at every z-position.
            The z-stage parameters (steps, step_um) are taken from the first
            channel; all channels are assumed to share the same z configuration.
        """
        first_config = channels_info[0][1]
        z_steps   = first_config.get('z_stage_steps', 1)
        z_step_um = first_config.get('z_stage_step_um', 0)
        ch_names  = [ch for ch, _, _ in channels_info]

        # Warn if channels have mismatched z-stage configurations
        for channel, config, _ in channels_info[1:]:
            ch_steps   = config.get('z_stage_steps', 1)
            ch_step_um = config.get('z_stage_step_um', 0)
            if ch_steps != z_steps or ch_step_um != z_step_um:
                self.log_message.emit(
                    f"[Interleaved Z-scan] WARNING: channel {channel} has a different "
                    f"z-stage config ({ch_steps} steps × {ch_step_um} µm) than channel "
                    f"{ch_names[0]} ({z_steps} steps × {z_step_um} µm). "
                    f"Using channel {ch_names[0]} parameters for all channels."
                )

        self.log_message.emit(
            f"[Interleaved Z-scan] Channels: {ch_names}, "
            f"{z_steps} steps × {z_step_um} µm"
        )

        z_moved = 0
        try:
            for z_idx in range(z_steps):
                if not self._is_running:
                    break
                if z_idx > 0:
                    asi_units = int(round(z_step_um * 10))
                    self.stage.move_relative_axis('Z', asi_units)
                    self.stage.wait_for_device()
                    z_moved += z_step_um
                self.log_message.emit(
                    f"[Z-step {z_idx}/{z_steps - 1}] offset: {z_moved:.1f} µm"
                )
                for channel, config, vol_index in channels_info:
                    if not self._is_running:
                        break
                    self._acquire_single_volume(
                        channel, config, vol_index, z_step=z_idx,
                    )
        finally:
            if z_moved != 0:
                self.log_message.emit(
                    "[Interleaved Z-scan] Returning stage to starting Z..."
                )
                self.stage.move_relative_axis('Z', -int(round(z_moved * 10)))
                self.stage.wait_for_device()

    # -------------------------------------------------------------------------
    # Volume acquisition (with optional Z-stage stepping)
    # -------------------------------------------------------------------------

    def _acquire_volume(self, channel, config, vol_index):
        """Adquiere uno o más volúmenes galvo-scaneados en diferentes posiciones Z.

        Si el Z-stage scan está habilitado, mueve el escenario por los pasos
        configurados y adquiere un volumen en cada posición. El escenario
        siempre vuelve a la posición inicial, incluso si ocurre un error.
        """
        z_enabled  = config.get('z_stage_enabled', False)
        z_steps    = config.get('z_stage_steps', 1)
        z_step_um  = config.get('z_stage_step_um', 0)

        if z_enabled and z_steps > 1 and self.stage is not None:
            self.log_message.emit(
                f"[{channel}] Z-stage scan: {z_steps} steps, "
                f"{z_step_um} um/step"
            )
            z_moved = 0
            try:
                for z_idx in range(z_steps):
                    if not self._is_running:
                        break
                    if z_idx > 0:
                        asi_units = int(round(z_step_um * 10))
                        self.stage.move_relative_axis('Z', asi_units)
                        self.stage.wait_for_device()
                        z_moved += z_step_um
                    self.log_message.emit(
                        f"[{channel}] Z-step {z_idx}/{z_steps - 1} "
                        f"(offset: {z_moved:.1f} um)"
                    )
                    self._acquire_single_volume(
                        channel, config, vol_index, z_step=z_idx,
                    )
            finally:
                if z_moved != 0:
                    self.log_message.emit(
                        f"[{channel}] Returning stage to starting Z..."
                    )
                    self.stage.move_relative_axis(
                        'Z', -int(round(z_moved * 10)),
                    )
                    self.stage.wait_for_device()
        else:
            self._acquire_single_volume(channel, config, vol_index)

    # -------------------------------------------------------------------------
    # Single galvo-scanned volume
    # -------------------------------------------------------------------------

    def _acquire_single_volume(self, channel, config, vol_index, z_step=None):
        """Adquiere un único volumen galvo-scaneado."""
        try:
            ch_camera = HARDWARE_CONFIG['channel_camera_trigger']
            ch_galvo  = HARDWARE_CONFIG['channel_galvo']

            # Debug: mostrar flags de guardado
            self.log_message.emit(
                f"[{channel}] Save flags → raw={config['save_raw']}, "
                f"crop={config['save_crop']}, deskew={config['save_deskew']}"
            )

            # Mover la rueda de filtros
            filter_position = config.get('filter_position', 1)
            if self.filter_wheel is not None:
                self.log_message.emit(
                    f"[{channel}] Moving filter to position {filter_position}..."
                )
                if self.filter_wheel.move_to_position(filter_position):
                    self.log_message.emit(
                        f"[{channel}] Filter: {self.filter_wheel.get_current_filter()}"
                    )
                else:
                    self.log_message.emit(
                        f"[{channel}] WARNING: Could not move filter wheel"
                    )

            exposure_us   = config['exposure_ms'] * 1000
            scan_range_um = config['scan_range_um']
            save_raw      = config['save_raw']
            save_crop     = config['save_crop']
            save_deskew   = config['save_deskew']
            drift_enabled = config['drift_correction']
            image_path    = config['image_path']

            hstart = config.get('roi_hstart', 620)
            hend   = config.get('roi_hend',   1320)
            vstart = config.get('roi_vstart',  635)
            vend   = config.get('roi_vend',   1465)

            self.log_message.emit(
                f"[{channel}] ROI → hstart={hstart}, hend={hend}, "
                f"vstart={vstart}, vend={vend}"
            )

            # Sufijo del archivo: incluye el índice z si se usa Z-stage
            if z_step is not None:
                suffix = f'{channel}_v{vol_index}_z{z_step}'
            else:
                suffix = f'{channel}_{vol_index}'

            # Calcular parámetros del scan.
            # El paso está fijado por Nyquist (resolución_axial / 2); ver config.py.
            scan_step_um = self.galvo_step_um
            slices       = 1 + int(np.round(scan_range_um / scan_step_um))

            # Configurar cámara
            self.camera.apply_settings(
                num_images   = slices,
                exposure_us  = exposure_us,
                height_px    = self.height_px,
                width_px     = self.width_px,
                timestamp    = 'binary+ASCII',
                trigger      = 'external',
            )

            # Calcular timing del DAQ
            exposure_px = self.ao.s2p(exposure_us / 1e6)
            rolling_px  = self.ao.s2p(self.camera.rolling_time_us / 1e6)
            jitter_px   = rolling_px
            period_px   = max(exposure_px, rolling_px) + jitter_px

            # Voltajes del galvo
            galvo_v_per_um = DEFAULT_CONFIG['galvo_volts_per_um']
            galvo_volts    = galvo_v_per_um * scan_range_um
            galvo_voltages = np.linspace(-galvo_volts / 2, galvo_volts / 2, slices)

            # Construir waveform de voltajes
            voltages = []
            for s in range(slices):
                volt_period = np.zeros(
                    (period_px, self.ao.num_channels), 'float64'
                )
                volt_period[:rolling_px, ch_camera] = 3.3
                volt_period[:, ch_galvo]            = galvo_voltages[s]
                voltages.append(volt_period)
            voltages = np.concatenate(voltages, axis=0)

            # ADQUIRIR
            laser = self.lasers[channel]
            laser.set_shutter("OPEN")

            self.ao.write_voltages(voltages)
            images = np.zeros(
                (slices, self.camera.height_px, self.camera.width_px),
                'uint16',
            )
            self.ao.play_voltages(block=False)
            self.camera.record_to_memory(images, software_trigger=False)

            laser.set_shutter("CLOSE")

            # Guardar raw
            if save_raw:
                images_raw = images.transpose(0, 2, 1)
                raw_path   = os.path.join(image_path, f'data_raw_{suffix}.tif')
                imwrite(raw_path, images_raw, imagej=True, metadata={'axes': 'ZYX'})
                self.log_message.emit(f"[{channel}] Saved: {raw_path}")

            # Aplicar crop
            if save_crop:
                if hend > hstart and vend > vstart:
                    images    = images[:, vstart:vend, hstart:hend]
                    crop_path = os.path.join(image_path, f'data_crop_{suffix}.tif')
                    imwrite(crop_path, images, imagej=True, metadata={'axes': 'ZYX'})
                    self.log_message.emit(f"[{channel}] Saved: {crop_path}")
                else:
                    self.log_message.emit(
                        f"[{channel}] WARNING: save_crop=True pero el ROI no es válido "
                        f"(hstart={hstart}, hend={hend}, vstart={vstart}, vend={vend}). "
                        f"Usá el botón 'Create ROI' en el live view para definir el ROI."
                    )

            # Deskew + max projection + drift correction
            every       = config.get('drift_correction_every', 1)
            needs_drift = drift_enabled and (vol_index % every == 0)
            if save_deskew or needs_drift:
                if not save_deskew:
                    self.log_message.emit(
                        f"[{channel}] Deskewing for drift correction (not saved)..."
                    )
                else:
                    self.log_message.emit(
                        f"[{channel}] Iniciando deskew sobre imagen {images.shape}..."
                    )
                try:
                    images_for_deskew = np.flip(np.transpose(images, (0, 2, 1)), axis=0)
                    deskewed = deskew(
                        data         = images_for_deskew,
                        theta        = np.rad2deg(self.tilt),
                        distance     = scan_step_um,
                        pixel_size   = self.sample_px_um,
                        z_downsample = 2,
                        output_dtype = 'uint16',
                    )

                    if save_deskew:
                        deskew_path = os.path.join(
                            image_path, f'data_deskew_{suffix}.tif'
                        )
                        imwrite(deskew_path, deskewed, imagej=True, metadata={'axes': 'ZYX'})
                        self.log_message.emit(f"[{channel}] Saved: {deskew_path}")

                        maxz      = max_projection_z(deskewed)
                        maxz_path = os.path.join(image_path, f'data_maxz_{suffix}.tif')
                        imwrite(maxz_path, maxz, imagej=True, metadata={'axes': 'YX'})
                        self.log_message.emit(f"[{channel}] Saved: {maxz_path}")

                    if needs_drift:
                        self._apply_drift_correction(
                            channel,
                            deskewed,
                            drift_type      = config.get('drift_correction_type', '3D full volume'),
                            ref_update_every= config.get('drift_correction_ref_update', 0),
                        )
                except Exception as deskew_err:
                    self.log_message.emit(
                        f"[{channel}] ERROR en deskew: {deskew_err}. "
                        f"Si el error es de memoria, asegurate de usar 'Create ROI' "
                        f"para reducir el tamaño de la imagen antes del deskew."
                    )

            self.log_message.emit(f"[{channel}] Volume {suffix} completed")

        except Exception as e:
            self.log_message.emit(f"[{channel}] ERROR in volume {suffix}: {e}")

    # -------------------------------------------------------------------------
    # Drift correction
    # -------------------------------------------------------------------------

    def _apply_drift_correction(self, channel, deskewed,
                                drift_type='3D full volume', ref_update_every=0):
        """Corrección de drift por cross-correlación de fase.

        drift_type:
            '3D full volume'   — cross-correlación sobre el volumen 3D completo.
            '2D max-projection'— 3 proyecciones ortogonales (XY, XZ, YZ) sobre el
                                 volumen completo en Z. dZ estimado exclusivamente
                                 por promedio ponderado de las correlaciones XZ e YZ.
        ref_update_every:
            0  — referencia fija (primer volumen).
            N  — actualiza la referencia cada N correcciones exitosas.
        """
        FAILSAFE_THRESHOLD = 0.4  # saltea si el frame tiene <40% del brillo de referencia

        def _preprocess(arr):
            """Normalización percentil [p1, p99.9] → [0, 1] (funciona en 2D y 3D)."""
            a = arr.astype(np.float32)
            p1  = np.percentile(a, 1)
            p99 = np.percentile(a, 99.9)
            rng = float(p99 - p1)
            return np.clip((a - p1) / max(rng, 1e-5), 0.0, 1.0)

        try:
            use_2d = (drift_type == '2D max-projection')

            # Si el método cambió respecto a la referencia almacenada, resetear.
            ref = self.reference_images.get(channel)
            if ref is not None:
                stored_2d = isinstance(ref, dict)
                if stored_2d != use_2d:
                    self.reference_images[channel]        = None
                    self.drift_correction_counts[channel] = 0
                    self.log_message.emit(
                        f"[{channel}] Drift method changed — reference reset"
                    )

            # ----------------------------------------------------------------
            # Modo 2D: 3 proyecciones ortogonales sobre el volumen Z completo.
            # dX, dY desde correlación de la proyección XY.
            # dZ desde promedio ponderado de correlaciones XZ e YZ
            # (peso = 1/error; menor error → más confianza).
            # Failsafe: si el frame es muy oscuro se omite la corrección.
            # ----------------------------------------------------------------
            if use_2d:
                brightness_curr = float(np.percentile(deskewed, 99.9))

                ref = self.reference_images[channel]

                # Failsafe: skip si el frame cayó abruptamente (fotoblanqueo/parpadeo)
                if ref is not None:
                    if brightness_curr < ref['brightness'] * FAILSAFE_THRESHOLD:
                        self.log_message.emit(
                            f"[{channel}] Drift FAILSAFE: frame oscuro "
                            f"({brightness_curr:.0f} < {ref['brightness'] * FAILSAFE_THRESHOLD:.0f}), "
                            f"corrección omitida"
                        )
                        return

                # Proyecciones MAX sobre TODO el rango Z
                proj_xy = _preprocess(np.max(deskewed, axis=0))  # (Y, X)
                proj_xz = _preprocess(np.max(deskewed, axis=1))  # (Z, X)
                proj_yz = _preprocess(np.max(deskewed, axis=2))  # (Z, Y)

                if ref is None:
                    self.reference_images[channel] = {
                        'xy': proj_xy, 'xz': proj_xz, 'yz': proj_yz,
                        'brightness': brightness_curr,
                    }
                    self.drift_correction_counts[channel] = 0
                    self.log_message.emit(
                        f"[{channel}] Drift reference set (2D max-projection)"
                    )
                    return

                # XY: shift[0]=dY, shift[1]=dX
                shift_xy, _,      _ = phase_cross_correlation(ref['xy'], proj_xy, upsample_factor=10)
                shift_xz, err_xz, _ = phase_cross_correlation(ref['xz'], proj_xz, upsample_factor=10)
                shift_yz, err_yz, _ = phase_cross_correlation(ref['yz'], proj_yz, upsample_factor=10)

                dy_px = float(shift_xy[0])
                dx_px = float(shift_xy[1])

                # dZ: promedio ponderado de ambas proyecciones axiales
                w_xz  = 1.0 / (float(err_xz) + 1e-5)
                w_yz  = 1.0 / (float(err_yz) + 1e-5)
                dz_px = (float(shift_xz[0]) * w_xz + float(shift_yz[0]) * w_yz) / (w_xz + w_yz)

                self.log_message.emit(
                    f"[{channel}] dZ={dz_px:+.2f}px "
                    f"(w_xz={w_xz:.1f}, w_yz={w_yz:.1f})"
                )

                new_ref_2d = {
                    'xy': proj_xy, 'xz': proj_xz, 'yz': proj_yz,
                    'brightness': brightness_curr,
                }

            # ----------------------------------------------------------------
            # Modo 3D: cross-correlación sobre el volumen completo
            # ----------------------------------------------------------------
            else:
                filtered = _preprocess(deskewed)

                if self.reference_images[channel] is None:
                    self.reference_images[channel]        = filtered
                    self.drift_correction_counts[channel] = 0
                    self.log_message.emit(
                        f"[{channel}] Drift reference set (3D full volume)"
                    )
                    return

                shift, _, _ = phase_cross_correlation(
                    self.reference_images[channel], filtered, upsample_factor=10
                )
                dz_px = float(shift[0])
                dx_px = float(shift[1])
                dy_px = float(shift[2])

            # ----------------------------------------------------------------
            # Convertir píxeles → µm y mover stage
            # ----------------------------------------------------------------
            dz_um = dz_px * DEFAULT_CONFIG['drift_z_um_per_px'] * (-1)
            dx_um = dx_px * DEFAULT_CONFIG['drift_xy_um_per_px']
            dy_um = dy_px * DEFAULT_CONFIG['drift_xy_um_per_px']

            self.log_message.emit(
                f"[{channel}] Drift ({drift_type}): "
                f"Z={dz_um:.3f}µm, Y={dy_um:.3f}µm, X={dx_um:.3f}µm"
            )

            if abs(dz_um) > self.drift_threshold and abs(dz_um) < self.safety_z:
                dz_um_safe = min(abs(dz_um), self.safety_z) * np.sign(dz_um)
                self.stage.move_relative_axis('Z', int(dz_um_safe * 10))
                self.log_message.emit(f"[{channel}] Correcting Z: {dz_um_safe:.3f}µm")

            if abs(dy_um) > self.drift_threshold and abs(dy_um) < self.safety_xy:
                dy_um_safe = min(abs(dy_um), self.safety_xy) * np.sign(dy_um)
                self.stage.move_relative_axis('Y', int(dy_um_safe * 10))
                self.log_message.emit(f"[{channel}] Correcting Y: {dy_um_safe:.3f}µm")

            if abs(dx_um) > self.drift_threshold and abs(dx_um) < self.safety_xy:
                dx_um_safe = min(abs(dx_um), self.safety_xy) * np.sign(dx_um)
                self.stage.move_relative_axis('X', int(dx_um_safe * 10))
                self.log_message.emit(f"[{channel}] Correcting X: {dx_um_safe:.3f}µm")

            # ----------------------------------------------------------------
            # Referencia adaptativa
            # ----------------------------------------------------------------
            self.drift_correction_counts[channel] += 1
            if ref_update_every > 0 and \
               self.drift_correction_counts[channel] % ref_update_every == 0:
                self.reference_images[channel] = new_ref_2d if use_2d else filtered
                self.log_message.emit(
                    f"[{channel}] Drift reference updated "
                    f"(correction #{self.drift_correction_counts[channel]})"
                )

        except Exception as e:
            self.log_message.emit(f"[{channel}] Error in drift correction: {e}")

    def request_stop(self):
        """Solicita al scheduler que detenga la adquisición."""
        self.log_message.emit("[Scheduler] Stop requested")
        self._is_running = False
