"""PCO.edge 4.2 CamLink sCMOS Camera Controller.

Provides basic control for the PCO.edge 4.2 CamLink camera via the
SC2_Cam DLL (PCO SDK). Many more commands are available in the SDK
and have not been implemented here.
"""

import ctypes as C
import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)

# --- DLL setup ---
_DLL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PCO')
os.add_dll_directory(_DLL_DIR)

try:
    _dll = C.oledll.LoadLibrary(os.path.join(_DLL_DIR, 'SC2_Cam'))
except OSError as exc:
    raise ImportError(
        f"Could not load 'SC2_Cam.dll' from '{_DLL_DIR}'. Ensure that "
        f"'SC2_Cam.dll' and 'sc2_cl_me4.dll' are in the PCO subdirectory."
    ) from exc


class CameraError(RuntimeError):
    """Exception raised when the PCO camera reports an error."""


# Error text retrieval
_dll.get_error_text = _dll.PCO_GetErrorText
_dll.get_error_text.argtypes = [C.c_uint32, C.c_char_p, C.c_uint32]


def _check_error(error_code):
    """Callback used as restype for DLL functions to check PCO errors."""
    if error_code == 0:
        return 0
    dwlen = 1000
    error_description = C.c_char_p(dwlen * b'')
    _dll.get_error_text(error_code, error_description, dwlen)
    raise CameraError(error_description.value.decode('ascii'))


# --- PCO buffer list structure ---

class PcoBuflist(C.Structure):
    """Structure matching the PCO_Buflist layout expected by the SDK."""
    _fields_ = [
        ("SBufNr", C.c_int16),
        ("reserved", C.c_uint16),
        ("dwStatusDll", C.c_uint32),
        ("dwStatusDrv", C.c_uint32),
    ]


# --- DLL function wrappers ---

_dll.reboot_camera = _dll.PCO_RebootCamera
_dll.reboot_camera.argtypes = [C.c_void_p]

_dll.reset_dll = _dll.PCO_ResetLib
_dll.reset_dll.restype = _check_error

_dll.open_camera = _dll.PCO_OpenCamera
_dll.open_camera.argtypes = [C.POINTER(C.c_void_p), C.c_uint16]
_dll.open_camera.restype = _check_error

_dll.get_camera_name = _dll.PCO_GetCameraName
_dll.get_camera_name.argtypes = [C.c_void_p, C.c_char_p, C.c_uint16]
_dll.get_camera_name.restype = _check_error

_dll.reset_settings_to_default = _dll.PCO_ResetSettingsToDefault
_dll.reset_settings_to_default.argtypes = [C.c_void_p]
_dll.reset_settings_to_default.restype = _check_error

_dll.get_camera_health = _dll.PCO_GetCameraHealthStatus
_dll.get_camera_health.argtypes = [
    C.c_void_p,
    C.POINTER(C.c_uint32),
    C.POINTER(C.c_uint32),
    C.POINTER(C.c_uint32),
]
_dll.get_camera_health.restype = _check_error

_dll.get_temperature = _dll.PCO_GetTemperature
_dll.get_temperature.argtypes = [
    C.c_void_p,
    C.POINTER(C.c_int16),
    C.POINTER(C.c_int16),
    C.POINTER(C.c_int16),
]
_dll.get_temperature.restype = _check_error

_dll.get_sensor_format = _dll.PCO_GetSensorFormat
_dll.get_sensor_format.argtypes = [C.c_void_p, C.POINTER(C.c_uint16)]
_dll.get_sensor_format.restype = _check_error

_dll.set_sensor_format = _dll.PCO_SetSensorFormat
_dll.set_sensor_format.argtypes = [C.c_void_p, C.c_uint16]
_dll.set_sensor_format.restype = _check_error

_dll.get_acquire_mode = _dll.PCO_GetAcquireMode
_dll.get_acquire_mode.argtypes = [C.c_void_p, C.POINTER(C.c_uint16)]
_dll.get_acquire_mode.restype = _check_error

_dll.set_acquire_mode = _dll.PCO_SetAcquireMode
_dll.set_acquire_mode.argtypes = [C.c_void_p, C.c_uint16]
_dll.set_acquire_mode.restype = _check_error

_dll.get_pixel_rate = _dll.PCO_GetPixelRate
_dll.get_pixel_rate.argtypes = [C.c_void_p, C.POINTER(C.c_uint32)]
_dll.get_pixel_rate.restype = _check_error

_dll.set_pixel_rate = _dll.PCO_SetPixelRate
_dll.set_pixel_rate.argtypes = [C.c_void_p, C.c_uint32]
_dll.set_pixel_rate.restype = _check_error

_dll.get_storage_mode = _dll.PCO_GetStorageMode
_dll.get_storage_mode.argtypes = [C.c_void_p, C.POINTER(C.c_uint16)]
_dll.get_storage_mode.restype = _check_error

_dll.set_storage_mode = _dll.PCO_SetStorageMode
_dll.set_storage_mode.argtypes = [C.c_void_p, C.c_uint16]
_dll.set_storage_mode.restype = _check_error

_dll.get_recorder_submode = _dll.PCO_GetRecorderSubmode
_dll.get_recorder_submode.argtypes = [C.c_void_p, C.POINTER(C.c_uint16)]
_dll.get_recorder_submode.restype = _check_error

_dll.set_recorder_submode = _dll.PCO_SetRecorderSubmode
_dll.set_recorder_submode.argtypes = [C.c_void_p, C.c_uint16]
_dll.set_recorder_submode.restype = _check_error

_dll.get_timestamp_mode = _dll.PCO_GetTimestampMode
_dll.get_timestamp_mode.argtypes = [C.c_void_p, C.POINTER(C.c_uint16)]
_dll.get_timestamp_mode.restype = _check_error

_dll.set_timestamp_mode = _dll.PCO_SetTimestampMode
_dll.set_timestamp_mode.argtypes = [C.c_void_p, C.c_uint16]
_dll.set_timestamp_mode.restype = _check_error

_dll.get_trigger_mode = _dll.PCO_GetTriggerMode
_dll.get_trigger_mode.argtypes = [C.c_void_p, C.POINTER(C.c_uint16)]
_dll.get_trigger_mode.restype = _check_error

_dll.set_trigger_mode = _dll.PCO_SetTriggerMode
_dll.set_trigger_mode.argtypes = [C.c_void_p, C.c_uint16]
_dll.set_trigger_mode.restype = _check_error

_dll.force_trigger = _dll.PCO_ForceTrigger
_dll.force_trigger.argtypes = [C.c_void_p, C.POINTER(C.c_uint16)]
_dll.force_trigger.restype = _check_error

_dll.get_delay_exposure_time = _dll.PCO_GetDelayExposureTime
_dll.get_delay_exposure_time.argtypes = [
    C.c_void_p,
    C.POINTER(C.c_uint32),
    C.POINTER(C.c_uint32),
    C.POINTER(C.c_uint16),
    C.POINTER(C.c_uint16),
]
_dll.get_delay_exposure_time.restype = _check_error

_dll.set_delay_exposure_time = _dll.PCO_SetDelayExposureTime
_dll.set_delay_exposure_time.argtypes = [
    C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint16, C.c_uint16,
]
_dll.set_delay_exposure_time.restype = _check_error

_dll.get_roi = _dll.PCO_GetROI
_dll.get_roi.argtypes = [
    C.c_void_p,
    C.POINTER(C.c_uint16),
    C.POINTER(C.c_uint16),
    C.POINTER(C.c_uint16),
    C.POINTER(C.c_uint16),
]
_dll.get_roi.restype = _check_error

_dll.set_roi = _dll.PCO_SetROI
_dll.set_roi.argtypes = [
    C.c_void_p, C.c_uint16, C.c_uint16, C.c_uint16, C.c_uint16,
]
_dll.set_roi.restype = _check_error

_dll.arm_camera = _dll.PCO_ArmCamera
_dll.arm_camera.argtypes = [C.c_void_p]
_dll.arm_camera.restype = _check_error

_dll.get_sizes = _dll.PCO_GetSizes
_dll.get_sizes.argtypes = [
    C.c_void_p,
    C.POINTER(C.c_uint16),
    C.POINTER(C.c_uint16),
    C.POINTER(C.c_uint16),
    C.POINTER(C.c_uint16),
]
_dll.get_sizes.restype = _check_error

_dll.allocate_buffer = _dll.PCO_AllocateBuffer
_dll.allocate_buffer.argtypes = [
    C.c_void_p,
    C.POINTER(C.c_int16),
    C.c_uint32,
    C.POINTER(C.POINTER(C.c_uint16)),
    C.POINTER(C.c_void_p),
]
_dll.allocate_buffer.restype = _check_error

_dll.set_image_parameters = _dll.PCO_SetImageParameters
_dll.set_image_parameters.argtypes = [
    C.c_void_p, C.c_uint16, C.c_uint16, C.c_uint32,
    C.POINTER(C.c_void_p), C.c_int32,
]
_dll.set_image_parameters.restype = _check_error

_dll.set_recording_state = _dll.PCO_SetRecordingState
_dll.set_recording_state.argtypes = [C.c_void_p, C.c_uint16]
_dll.set_recording_state.restype = _check_error

_dll.add_buffer = _dll.PCO_AddBufferEx
_dll.add_buffer.argtypes = [
    C.c_void_p, C.c_uint32, C.c_uint32, C.c_int16,
    C.c_uint16, C.c_uint16, C.c_uint16,
]
_dll.add_buffer.restype = _check_error

_dll.get_buffer_status = _dll.PCO_GetBufferStatus
_dll.get_buffer_status.argtypes = [
    C.c_void_p, C.c_int16,
    C.POINTER(C.c_uint32), C.POINTER(C.c_uint32),
]
_dll.get_buffer_status.restype = _check_error

_dll.wait_for_buffer = _dll.PCO_WaitforBuffer
_dll.wait_for_buffer.argtypes = [
    C.c_void_p, C.c_int, C.POINTER(PcoBuflist), C.c_int,
]
_dll.wait_for_buffer.restype = _check_error

_dll.cancel_images = _dll.PCO_CancelImages
_dll.cancel_images.argtypes = [C.c_void_p]
_dll.cancel_images.restype = _check_error

_dll.free_buffer = _dll.PCO_FreeBuffer
_dll.free_buffer.argtypes = [C.c_void_p, C.c_int16]
_dll.free_buffer.restype = _check_error

_dll.close_camera = _dll.PCO_CloseCamera
_dll.close_camera.argtypes = [C.c_void_p]
_dll.close_camera.restype = _check_error


# --- Mode lookup tables ---

_SENSOR_FORMATS = {0: "standard", 1: "extended"}
_SENSOR_FORMATS_INV = {v: k for k, v in _SENSOR_FORMATS.items()}

_ACQUIRE_MODES = {0: "auto", 1: "external", 2: "external_modulate"}
_ACQUIRE_MODES_INV = {v: k for k, v in _ACQUIRE_MODES.items()}

_STORAGE_MODES = {0: "recorder", 1: "FIFO_buffer"}
_STORAGE_MODES_INV = {v: k for k, v in _STORAGE_MODES.items()}

_RECORDER_SUBMODES = {0: "sequence", 1: "ring_buffer"}
_RECORDER_SUBMODES_INV = {v: k for k, v in _RECORDER_SUBMODES.items()}

_TIMESTAMP_MODES = {0: "off", 1: "binary", 2: "binary+ASCII"}
_TIMESTAMP_MODES_INV = {v: k for k, v in _TIMESTAMP_MODES.items()}

_TRIGGER_MODES = {0: "auto", 1: "software", 2: "external", 3: "external_exposure"}
_TRIGGER_MODES_INV = {v: k for k, v in _TRIGGER_MODES.items()}

_TIMEBASE_TO_US = {0: 1e-3, 1: 1, 2: 1e3}  # {0: ns, 1: us, 2: ms}

_PIXEL_RATES = {
    95333333: 27.77,   # line time in us
    272250000: 9.76,
}

# Image size constraints
_HEIGHT_STEP = 1
_WIDTH_STEP = 20
_MIN_HEIGHT, _MIN_WIDTH = 10, 40
_MAX_HEIGHT, _MAX_WIDTH = 2048, 2060
_EXPECTED_CAMERA_NAME = b'pco.edge rolling shutter 4.2'
_CAMERA_NAME_LEN = 40
_BUFFER_EVENT_SET = 0xE0008000
_DMA_ERROR = 0x80332028


def legalize_image_size(
    height_px: "int | str" = 'max',
    width_px: "int | str" = 'max',
    name: str = 'PCO.edge4.2',
) -> tuple:
    """Compute the nearest legal, centered image size for the camera chip.

    Parameters
    ----------
    height_px : int or 'min' or 'max'
        Requested image height in pixels.
    width_px : int or 'min' or 'max'
        Requested image width in pixels.
    name : str
        Device name for log messages.

    Returns
    -------
    tuple
        (height_px, width_px, roi_px) where roi_px is a dict with
        keys 'left', 'right', 'top', 'bottom'.
    """
    logger.info("%s: requested image size = %s x %s (h x w)", name, height_px, width_px)

    if height_px == 'min':
        height_px = _MIN_HEIGHT
    elif height_px == 'max':
        height_px = _MAX_HEIGHT
    if width_px == 'min':
        width_px = _MIN_WIDTH
    elif width_px == 'max':
        width_px = _MAX_WIDTH

    if not isinstance(height_px, int) or not isinstance(width_px, int):
        raise TypeError("height_px and width_px must be int, 'min', or 'max'.")
    if not _MIN_HEIGHT <= height_px <= _MAX_HEIGHT:
        raise ValueError(
            f"height_px={height_px} out of range [{_MIN_HEIGHT}, {_MAX_HEIGHT}]."
        )
    if not _MIN_WIDTH <= width_px <= _MAX_WIDTH:
        raise ValueError(
            f"width_px={width_px} out of range [{_MIN_WIDTH}, {_MAX_WIDTH}]."
        )

    num_height_steps = height_px // _HEIGHT_STEP
    num_width_steps = width_px // _WIDTH_STEP
    if num_height_steps % 2 != 0:
        num_height_steps += 1  # must be even for chip
    if num_width_steps % 2 == 0:
        num_width_steps += 1   # must be odd for chip

    height_px = _HEIGHT_STEP * num_height_steps
    width_px = _WIDTH_STEP * num_width_steps

    ud_center = _MAX_HEIGHT / 2
    lr_center = _MAX_WIDTH / 2
    left = int(lr_center - (width_px / 2)) + 1
    right = int(lr_center + (width_px / 2))
    top = int(ud_center - (height_px / 2)) + 1
    bottom = int(ud_center + (height_px / 2))
    roi_px = {'left': left, 'right': right, 'top': top, 'bottom': bottom}

    logger.info(
        "%s: legal image size = %i x %i (h x w), roi = %s",
        name, height_px, width_px, roi_px,
    )
    return height_px, width_px, roi_px


class Camera:
    """Controller for the PCO.edge 4.2 CamLink sCMOS camera.

    Parameters
    ----------
    name : str
        A descriptive name for this camera instance.
    """

    def __init__(self, name: str = 'PCO.edge4.2_cl'):
        self.name = name
        logger.info("%s: opening...", self.name)

        self.handle = C.c_void_p(0)
        try:
            _dll.open_camera(self.handle, 0)
        except OSError:
            logger.error(
                "%s: failed to open. Check that the camera is on, "
                "connected, and CamWare is not running.", self.name,
            )
            raise
        if self.handle.value is None:
            raise CameraError(f"{self.name}: camera handle is null after open.")

        camera_name = C.c_char_p(_CAMERA_NAME_LEN * b' ')
        _dll.get_camera_name(self.handle, camera_name, _CAMERA_NAME_LEN)
        if camera_name.value != _EXPECTED_CAMERA_NAME:
            raise CameraError(
                f"{self.name}: unexpected camera name '{camera_name.value}'. "
                f"Expected '{_EXPECTED_CAMERA_NAME}'."
            )

        self._num_buffers = 16
        self._armed = False
        self._disarm()
        self._reset_settings_to_default()
        self._get_health_status(check=True)
        self._get_temperature()
        self._set_sensor_format('standard')
        self._set_acquire_mode('auto')
        self._set_pixel_rate(272250000)
        self._set_storage_mode('recorder')
        self._set_recorder_submode('ring_buffer')
        self._set_timestamp_mode('off')
        self._set_trigger_mode('external')
        self._set_exposure_time_us(100)
        roi = legalize_image_size('max', 'max', self.name)[2]
        self._set_roi(roi)
        self._get_image_size()
        self.num_images = 1
        logger.info("%s: open and ready.", self.name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # --- Internal helpers ---

    def _reboot(self, polling_time_s: float = 0.2, timeout_s: float = 15.0):
        """Reboot the camera and wait for it to become available again."""
        logger.info("%s: rebooting...", self.name)
        _dll.reboot_camera(self.handle)
        _dll.close_camera(self.handle)
        t0 = time.perf_counter()
        while True:
            try:
                _dll.reset_dll()
                time.sleep(polling_time_s)
                _dll.open_camera(self.handle, 0)
            except OSError:
                elapsed = time.perf_counter() - t0
                if elapsed > timeout_s:
                    raise
            else:
                break
        elapsed = time.perf_counter() - t0
        logger.info("%s: reboot done (%.2fs).", self.name, elapsed)

    def _reset_settings_to_default(self):
        """Reset all camera settings to factory defaults."""
        logger.debug("%s: resetting settings to default...", self.name)
        _dll.reset_settings_to_default(self.handle)

    def _get_health_status(self, check: bool = False) -> dict:
        """Read warning, error, and status bits from the camera.

        Parameters
        ----------
        check : bool
            If True, raises CameraError when warnings or errors are present.

        Returns
        -------
        dict
            Keys: 'warnings', 'errors', 'status'.
        """
        dw_warn, dw_err, dw_status = C.c_uint32(), C.c_uint32(), C.c_uint32()
        _dll.get_camera_health(self.handle, dw_warn, dw_err, dw_status)
        self.health = {
            'warnings': dw_warn.value,
            'errors': dw_err.value,
            'status': dw_status.value,
        }
        logger.debug("%s: health = %s", self.name, self.health)
        if check and (self.health['warnings'] != 0 or self.health['errors'] != 0):
            raise CameraError(
                f"{self.name}: health check failed. "
                f"Warnings={self.health['warnings']}, "
                f"Errors={self.health['errors']}."
            )
        return self.health

    def _get_temperature(self) -> dict:
        """Read CCD, camera body, and power supply temperatures.

        Returns
        -------
        dict
            Keys: 'ccd_degC', 'camera_degC', 'psu_degC'.
        """
        ccd, cam, psu = C.c_int16(), C.c_int16(), C.c_int16()
        _dll.get_temperature(self.handle, ccd, cam, psu)
        self.temperature = {
            'ccd_degC': ccd.value * 0.1,
            'camera_degC': cam.value,
            'psu_degC': psu.value,
        }
        logger.debug("%s: temperature = %s", self.name, self.temperature)
        return self.temperature

    def _get_sensor_format(self) -> str:
        """Get the current sensor format ('standard' or 'extended')."""
        w_sensor = C.c_uint16(0)
        _dll.get_sensor_format(self.handle, w_sensor)
        self.sensor_format = _SENSOR_FORMATS[w_sensor.value]
        logger.debug("%s: sensor format = %s", self.name, self.sensor_format)
        return self.sensor_format

    def _set_sensor_format(self, mode: str):
        """Set the sensor format."""
        if mode not in _SENSOR_FORMATS_INV:
            raise ValueError(f"Invalid sensor format '{mode}'.")
        _dll.set_sensor_format(self.handle, _SENSOR_FORMATS_INV[mode])
        if self._get_sensor_format() != mode:
            raise CameraError(f"{self.name}: failed to set sensor format to '{mode}'.")

    def _get_acquire_mode(self) -> str:
        """Get the current acquire mode."""
        w_mode = C.c_uint16(0)
        _dll.get_acquire_mode(self.handle, w_mode)
        self.acquire_mode = _ACQUIRE_MODES[w_mode.value]
        logger.debug("%s: acquire mode = %s", self.name, self.acquire_mode)
        return self.acquire_mode

    def _set_acquire_mode(self, mode: str):
        """Set the acquire mode."""
        if mode not in _ACQUIRE_MODES_INV:
            raise ValueError(f"Invalid acquire mode '{mode}'.")
        _dll.set_acquire_mode(self.handle, _ACQUIRE_MODES_INV[mode])
        if self._get_acquire_mode() != mode:
            raise CameraError(f"{self.name}: failed to set acquire mode to '{mode}'.")

    def _get_pixel_rate(self) -> int:
        """Get the current pixel rate in Hz."""
        dw_rate = C.c_uint32(0)
        _dll.get_pixel_rate(self.handle, dw_rate)
        self.pixel_rate = dw_rate.value
        if self.pixel_rate == 0:
            raise CameraError(f"{self.name}: pixel rate returned 0.")
        logger.debug("%s: pixel rate = %i Hz", self.name, self.pixel_rate)
        return self.pixel_rate

    def _set_pixel_rate(self, rate: int):
        """Set the pixel rate.

        Parameters
        ----------
        rate : int
            Pixel rate in Hz. Must be 95333333 or 272250000.
        """
        if rate not in _PIXEL_RATES:
            raise ValueError(
                f"Invalid pixel rate {rate}. "
                f"Allowed values: {list(_PIXEL_RATES.keys())}."
            )
        _dll.set_pixel_rate(self.handle, rate)
        if self._get_pixel_rate() != rate:
            raise CameraError(f"{self.name}: failed to set pixel rate to {rate}.")
        self.line_time_us = _PIXEL_RATES[rate]

    def _get_storage_mode(self) -> str:
        """Get the current storage mode."""
        w_mode = C.c_uint16(0)
        _dll.get_storage_mode(self.handle, w_mode)
        self.storage_mode = _STORAGE_MODES[w_mode.value]
        logger.debug("%s: storage mode = %s", self.name, self.storage_mode)
        return self.storage_mode

    def _set_storage_mode(self, mode: str):
        """Set the storage mode."""
        if mode not in _STORAGE_MODES_INV:
            raise ValueError(f"Invalid storage mode '{mode}'.")
        _dll.set_storage_mode(self.handle, _STORAGE_MODES_INV[mode])
        if self._get_storage_mode() != mode:
            raise CameraError(f"{self.name}: failed to set storage mode to '{mode}'.")

    def _get_recorder_submode(self) -> str:
        """Get the current recorder submode."""
        w_mode = C.c_uint16(0)
        _dll.get_recorder_submode(self.handle, w_mode)
        self.recorder_submode = _RECORDER_SUBMODES[w_mode.value]
        logger.debug("%s: recorder submode = %s", self.name, self.recorder_submode)
        return self.recorder_submode

    def _set_recorder_submode(self, mode: str):
        """Set the recorder submode."""
        if mode not in _RECORDER_SUBMODES_INV:
            raise ValueError(f"Invalid recorder submode '{mode}'.")
        _dll.set_recorder_submode(self.handle, _RECORDER_SUBMODES_INV[mode])
        if self._get_recorder_submode() != mode:
            raise CameraError(
                f"{self.name}: failed to set recorder submode to '{mode}'."
            )

    def _get_timestamp_mode(self) -> str:
        """Get the current timestamp mode."""
        w_mode = C.c_uint16(0)
        _dll.get_timestamp_mode(self.handle, w_mode)
        self.timestamp_mode = _TIMESTAMP_MODES[w_mode.value]
        logger.debug("%s: timestamp mode = %s", self.name, self.timestamp_mode)
        return self.timestamp_mode

    def _set_timestamp_mode(self, mode: str):
        """Set the timestamp mode."""
        if mode not in _TIMESTAMP_MODES_INV:
            raise ValueError(f"Invalid timestamp mode '{mode}'.")
        _dll.set_timestamp_mode(self.handle, _TIMESTAMP_MODES_INV[mode])
        if self._get_timestamp_mode() != mode:
            raise CameraError(
                f"{self.name}: failed to set timestamp mode to '{mode}'."
            )

    def _get_trigger_mode(self) -> str:
        """Get the current trigger mode."""
        w_mode = C.c_uint16(0)
        _dll.get_trigger_mode(self.handle, w_mode)
        self.trigger_mode = _TRIGGER_MODES[w_mode.value]
        logger.debug("%s: trigger mode = %s", self.name, self.trigger_mode)
        return self.trigger_mode

    def _set_trigger_mode(self, mode: str):
        """Set the trigger mode.

        Parameters
        ----------
        mode : str
            One of 'auto', 'software', 'external', 'external_exposure'.

            - 'auto': exposure starts automatically after readout.
            - 'software': exposure started by force trigger command.
            - 'external': sequence starts on rising/falling edge of
              trigger input.
            - 'external_exposure': exposure time defined by pulse length.
        """
        if mode not in _TRIGGER_MODES_INV:
            raise ValueError(f"Invalid trigger mode '{mode}'.")
        _dll.set_trigger_mode(self.handle, _TRIGGER_MODES_INV[mode])
        if self._get_trigger_mode() != mode:
            raise CameraError(f"{self.name}: failed to set trigger mode to '{mode}'.")

    def _force_trigger(self) -> bool:
        """Force a software trigger.

        Returns
        -------
        bool
            True if a new exposure was triggered, False otherwise.
        """
        if self.trigger_mode not in ('software', 'external'):
            raise CameraError(
                f"{self.name}: force trigger requires 'software' or "
                f"'external' trigger mode, current mode is '{self.trigger_mode}'."
            )
        w_triggered = C.c_uint16(0)
        _dll.force_trigger(self.handle, w_triggered)
        if w_triggered.value not in (0, 1):
            raise CameraError(
                f"{self.name}: unexpected trigger result {w_triggered.value}."
            )
        return bool(w_triggered.value)

    def _get_exposure_time_us(self) -> int:
        """Get the current delay and exposure times in microseconds.

        Returns
        -------
        int
            Exposure time in microseconds.
        """
        dw_delay = C.c_uint32(0)
        dw_exposure = C.c_uint32(0)
        w_tb_delay = C.c_uint16(0)
        w_tb_exposure = C.c_uint16(0)
        _dll.get_delay_exposure_time(
            self.handle, dw_delay, dw_exposure, w_tb_delay, w_tb_exposure,
        )
        self.delay_us = int(dw_delay.value * _TIMEBASE_TO_US[w_tb_delay.value])
        self.exposure_us = int(dw_exposure.value * _TIMEBASE_TO_US[w_tb_exposure.value])
        logger.debug(
            "%s: delay = %i us, exposure = %i us",
            self.name, self.delay_us, self.exposure_us,
        )
        return self.exposure_us

    def _set_exposure_time_us(self, exposure_us: int, delay_us: int = 0):
        """Set the exposure and delay times.

        Parameters
        ----------
        exposure_us : int
            Exposure time in microseconds (100 to 10,000,000).
        delay_us : int
            Delay time in microseconds (0 to 1,000,000).
        """
        if not isinstance(exposure_us, int) or not isinstance(delay_us, int):
            raise TypeError("exposure_us and delay_us must be int.")
        if not 100 <= exposure_us <= 10_000_000:
            raise ValueError(
                f"exposure_us={exposure_us} out of range [100, 10000000]."
            )
        if not 0 <= delay_us <= 1_000_000:
            raise ValueError(f"delay_us={delay_us} out of range [0, 1000000].")
        logger.debug(
            "%s: setting exposure = %i us, delay = %i us",
            self.name, exposure_us, delay_us,
        )
        _dll.set_delay_exposure_time(self.handle, delay_us, exposure_us, 1, 1)
        if self._get_exposure_time_us() != exposure_us:
            raise CameraError(
                f"{self.name}: failed to set exposure time to {exposure_us} us."
            )
        self.delay_us = delay_us

    def _get_roi(self) -> dict:
        """Get the current region of interest.

        Returns
        -------
        dict
            Keys: 'left', 'right', 'top', 'bottom'.
        """
        x0, y0, x1, y1 = (
            C.c_uint16(0), C.c_uint16(0), C.c_uint16(0), C.c_uint16(0),
        )
        _dll.get_roi(self.handle, x0, y0, x1, y1)
        self.roi_px = {
            'left': x0.value, 'right': x1.value,
            'top': y0.value, 'bottom': y1.value,
        }
        self.height_px = self.roi_px['bottom'] - self.roi_px['top'] + 1
        self.width_px = self.roi_px['right'] - self.roi_px['left'] + 1
        self.bytes_per_image = 2 * self.height_px * self.width_px  # 16-bit
        self.rolling_time_us = self.line_time_us * (self.height_px / 2)
        logger.debug("%s: roi = %s", self.name, self.roi_px)
        return self.roi_px

    def _set_roi(self, roi_px: dict):
        """Set the region of interest. Use legalize_image_size() first."""
        logger.debug("%s: setting roi = %s", self.name, roi_px)
        _dll.set_roi(
            self.handle,
            roi_px['left'], roi_px['top'],
            roi_px['right'], roi_px['bottom'],
        )
        if self._get_roi() != roi_px:
            raise CameraError(f"{self.name}: failed to set ROI to {roi_px}.")

    def _get_image_size(self) -> tuple:
        """Get the current image resolution.

        Returns
        -------
        tuple
            (height_px, width_px).
        """
        x_res, y_res, x_max, y_max = (
            C.c_uint16(0), C.c_uint16(0), C.c_uint16(0), C.c_uint16(0),
        )
        _dll.get_sizes(self.handle, x_res, y_res, x_max, y_max)
        height_px, width_px = y_res.value, x_res.value
        logger.debug(
            "%s: image size = %i x %i (h x w)", self.name, height_px, width_px,
        )
        return height_px, width_px

    def _disarm(self):
        """Stop recording and free all allocated buffers."""
        logger.debug("%s: disarming...", self.name)
        _dll.set_recording_state(self.handle, 0)
        _dll.cancel_images(self.handle)
        if self._armed:
            for i in range(self._num_buffers):
                _dll.free_buffer(self.handle, i)
        self._armed = False

    def _arm(self, num_buffers: int):
        """Arm the camera: allocate buffers and start the driver queue.

        Parameters
        ----------
        num_buffers : int
            Number of image buffers to allocate (1-16).
        """
        if self._armed:
            raise CameraError(f"{self.name}: camera is already armed.")
        if not 1 <= num_buffers <= 16:
            raise ValueError(f"num_buffers must be 1-16, got {num_buffers}.")

        logger.debug("%s: arming with %i buffers...", self.name, num_buffers)
        _dll.arm_camera(self.handle)
        if self._get_image_size() != (self.height_px, self.width_px):
            raise CameraError(f"{self.name}: image size mismatch after arming.")

        h_px, w_px = self.height_px, self.width_px
        self.buffers = []
        for i in range(num_buffers):
            buffer_index = C.c_int16(-1)
            self.buffers.append(np.zeros((h_px, w_px), 'uint16'))
            c_buffer = np.ctypeslib.as_ctypes(self.buffers[i])
            c_buffer_pointer = C.cast(c_buffer, C.POINTER(C.c_ushort))
            buffer_event = C.c_void_p(0)
            _dll.allocate_buffer(
                self.handle, buffer_index,
                self.bytes_per_image, c_buffer_pointer, buffer_event,
            )
            if buffer_index.value != i:
                raise CameraError(
                    f"{self.name}: buffer index mismatch "
                    f"(expected {i}, got {buffer_index.value})."
                )

        _dll.set_image_parameters(self.handle, w_px, h_px, 1, C.c_void_p(), 0)
        _dll.set_recording_state(self.handle, 1)

        self.added_buffers = []
        for i in range(num_buffers):
            _dll.add_buffer(self.handle, 0, 0, i, w_px, h_px, 16)
            self.added_buffers.append(i)

        self._armed = True
        self._num_buffers = num_buffers
        self.timeout_ms = int(1000 + 2 * 1e-3 * self.exposure_us)
        logger.debug("%s: armed.", self.name)

    # --- Public API ---

    def apply_settings(
        self,
        num_images: int = None,
        exposure_us: int = None,
        height_px: "int | str" = None,
        width_px: "int | str" = None,
        timestamp: str = None,
        trigger: str = None,
        num_buffers: int = None,
        timeout_ms: int = None,
        check_health: bool = True,
    ):
        """Configure the camera and arm it for acquisition.

        Parameters
        ----------
        num_images : int, optional
            Total number of images to record.
        exposure_us : int, optional
            Exposure time in microseconds (100 to 10,000,000).
        height_px : int or str, optional
            Image height in pixels (or 'min'/'max').
        width_px : int or str, optional
            Image width in pixels (or 'min'/'max').
        timestamp : str, optional
            Timestamp mode: 'off', 'binary', or 'binary+ASCII'.
        trigger : str, optional
            Trigger mode: 'auto', 'software', 'external', or
            'external_exposure'.
        num_buffers : int, optional
            Number of DMA buffers (1-16).
        timeout_ms : int, optional
            Buffer wait timeout in milliseconds.
        check_health : bool
            If True, performs a health check before arming.
        """
        logger.info("%s: applying settings...", self.name)
        if self._armed:
            self._disarm()
        if num_images is not None:
            if not isinstance(num_images, int):
                raise TypeError("num_images must be int.")
            self.num_images = num_images
        if exposure_us is not None:
            self._set_exposure_time_us(exposure_us)
        if height_px is not None or width_px is not None:
            if height_px is None:
                height_px = self.height_px
            if width_px is None:
                width_px = self.width_px
            roi_px = legalize_image_size(height_px, width_px, name=self.name)[2]
            self._set_roi(roi_px)
        if timestamp is not None:
            self._set_timestamp_mode(timestamp)
        if trigger is not None:
            self._set_trigger_mode(trigger)
        if check_health:
            self._get_health_status(check=True)
        if num_buffers is not None:
            self._num_buffers = num_buffers
        self._arm(self._num_buffers)
        if timeout_ms is not None:
            if not isinstance(timeout_ms, int):
                raise TypeError("timeout_ms must be int.")
            self.timeout_ms = timeout_ms
        logger.info("%s: settings applied.", self.name)

    def record_to_memory(
        self,
        allocated_memory: np.ndarray = None,
        software_trigger: bool = False,
    ) -> np.ndarray:
        """Record images into memory.

        Parameters
        ----------
        allocated_memory : np.ndarray, optional
            Pre-allocated uint16 array of shape
            (num_images, height_px, width_px). If None, a new array is
            created and returned.
        software_trigger : bool
            If True, force a software trigger for each image.

        Returns
        -------
        np.ndarray or None
            The recorded images if allocated_memory was None,
            otherwise None (images are written into the provided array).
        """
        if not self._armed:
            raise CameraError(
                f"{self.name}: camera not armed. Call apply_settings() first."
            )
        logger.info("%s: recording to memory...", self.name)

        h_px, w_px = self.height_px, self.width_px
        if allocated_memory is None:
            allocated_memory = np.zeros(
                (self.num_images, h_px, w_px), 'uint16',
            )
            output = allocated_memory
        else:
            if not isinstance(allocated_memory, np.ndarray):
                raise TypeError("allocated_memory must be a numpy ndarray.")
            if allocated_memory.dtype != np.uint16:
                raise TypeError("allocated_memory must have dtype uint16.")
            expected_shape = (self.num_images, h_px, w_px)
            if allocated_memory.shape != expected_shape:
                raise ValueError(
                    f"allocated_memory shape {allocated_memory.shape} "
                    f"does not match expected {expected_shape}."
                )
            output = None

        buflist = (PcoBuflist * 1)()
        for i in range(self.num_images):
            if software_trigger:
                if not self._force_trigger():
                    raise CameraError(f"{self.name}: software trigger failed.")
            buffer_index = self.added_buffers.pop(0)
            buflist[0].SBufNr = buffer_index
            try:
                _dll.wait_for_buffer(self.handle, 1, buflist, self.timeout_ms)
            except Exception:
                logger.error("%s: buffer timeout.", self.name)
                raise
            if buflist[0].dwStatusDll != _BUFFER_EVENT_SET:
                raise CameraError(
                    f"{self.name}: unexpected buffer DLL status "
                    f"0x{buflist[0].dwStatusDll:08X}."
                )
            if buflist[0].dwStatusDrv != 0:
                drv_status = buflist[0].dwStatusDrv
                detail = 'DMA error' if drv_status == _DMA_ERROR else (
                    f'0x{drv_status:08X}'
                )
                raise CameraError(
                    f"{self.name}: image transfer failed ({detail})."
                )
            allocated_memory[i, :, :] = self.buffers[buffer_index]
            _dll.add_buffer(self.handle, 0, 0, buffer_index, w_px, h_px, 16)
            self.added_buffers.append(buffer_index)

        logger.info("%s: recording complete.", self.name)
        return output

    # --- Live view API (pylablib-compatible interface) ---

    def set_exposure(self, exposure_s: float):
        """Set the exposure time in seconds.

        The new value is stored and applied on the next call to
        setup_acquisition().  This mirrors how pylablib cameras work.
        """
        self._pending_exposure_us = max(100, int(exposure_s * 1e6))

    def get_exposure(self) -> float:
        """Return the current exposure time in seconds."""
        return self.exposure_us / 1e6

    def setup_acquisition(self, nframes: int = 100):
        """Prepare the camera for free-running (live view) acquisition.

        Disarms if necessary, switches to auto-trigger mode, and
        applies any pending exposure change from set_exposure().
        """
        if self._armed:
            self._disarm()
        self._set_trigger_mode('auto')
        self._set_timestamp_mode('off')
        if hasattr(self, '_pending_exposure_us'):
            self._set_exposure_time_us(self._pending_exposure_us)
            del self._pending_exposure_us

    def start_acquisition(self):
        """Arm the camera and begin free-running acquisition.

        After this call, completed frames can be retrieved with
        read_newest_image().
        """
        if not self._armed:
            self._arm(self._num_buffers)

    def stop_acquisition(self):
        """Stop acquisition and disarm the camera."""
        if self._armed:
            self._disarm()

    def read_newest_image(self) -> "np.ndarray | None":
        """Return the most recent completed frame, or None.

        Drains all ready DMA buffers (non-blocking) and returns only
        the newest one.  Older frames are silently discarded so the
        live view never lags behind.
        """
        if not self._armed:
            return None

        newest = None
        w_px, h_px = self.width_px, self.height_px

        while self.added_buffers:
            idx = self.added_buffers[0]
            status_dll = C.c_uint32()
            status_drv = C.c_uint32()
            _dll.get_buffer_status(self.handle, idx, status_dll, status_drv)

            if status_dll.value != _BUFFER_EVENT_SET:
                break  # this buffer not ready yet — stop draining

            # Buffer ready: pop from queue
            self.added_buffers.pop(0)

            if status_drv.value == 0:
                newest = self.buffers[idx].copy()
            else:
                logger.warning(
                    "%s: buffer %d driver error 0x%08X",
                    self.name, idx, status_drv.value,
                )

            # Re-add buffer to the DMA queue for reuse
            _dll.add_buffer(self.handle, 0, 0, idx, w_px, h_px, 16)
            self.added_buffers.append(idx)

        return newest

    def snap(self, exposure_us: int = None) -> np.ndarray:
        """Capture a single image in auto-trigger mode.

        Parameters
        ----------
        exposure_us : int, optional
            Exposure time in microseconds. If None, the current value
            is used.

        Returns
        -------
        np.ndarray
            Single image of shape (height_px, width_px), dtype uint16.
        """
        if self._armed:
            self._disarm()
        if exposure_us is not None:
            self._set_exposure_time_us(exposure_us)
        self._set_trigger_mode('auto')
        self._set_timestamp_mode('off')
        self.num_images = 1
        self._arm(1)
        try:
            images = self.record_to_memory()
        finally:
            self._disarm()
        return images[0]

    def close(self):
        """Disarm the camera and release hardware resources."""
        self._disarm()
        logger.info("%s: closing...", self.name)
        _dll.close_camera(self.handle)
        logger.info("%s: closed.", self.name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    from tifffile import imwrite

    with Camera() as camera:
        camera.apply_settings(
            num_images=100,
            exposure_us=100,
            height_px=200,
            width_px=200,
            trigger="external",
        )
        images = camera.record_to_memory()
        imwrite('test03.tif', images, imagej=True)
        print("Done")
