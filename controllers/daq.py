"""National Instruments DAQ (PCIe-6363) Controller.

Provides analog output voltage control for a NI PCIe-6363 board
(32 AI 16-Bit 2 MS/s, 4 AO 2.86 MS/s, 48 DIO) via the NI-DAQmx C API.
"""

import ctypes as C
import logging

import numpy as np

logger = logging.getLogger(__name__)

# --- NI-DAQmx constants ---
DAQMX_VAL_RISING = 10280
DAQMX_VAL_FINITE_SAMPS = 10178
DAQMX_VAL_VOLTS = 10348
DAQMX_VAL_GROUP_BY_SCAN_NUMBER = 1

# --- DLL setup ---
try:
    _dll = C.cdll.LoadLibrary("nicaiu")
except OSError as exc:
    raise ImportError(
        "Could not load 'nicaiu.dll'. Ensure that NI-DAQmx drivers are "
        "installed and that the DLL is available in the system PATH."
    ) from exc


class DAQError(RuntimeError):
    """Exception raised when the NI-DAQmx driver reports an error."""


def _check_error(error_code):
    """Callback used as restype for DLL functions to check NI-DAQ errors."""
    if error_code != 0:
        num_bytes = _dll.DAQmxGetExtendedErrorInfo(None, 0)
        error_buffer = (C.c_char * num_bytes)()
        _dll.DAQmxGetExtendedErrorInfo(error_buffer, num_bytes)
        msg = error_buffer.value.decode('ascii')
        raise DAQError(f"NI-DAQ error code {error_code}: {msg}")
    return error_code


# Wrap each DLL call with proper argtypes / restype
_dll.create_task = _dll.DAQmxCreateTask
_dll.create_task.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
_dll.create_task.restype = _check_error

_dll.create_ao_voltage_channel = _dll.DAQmxCreateAOVoltageChan
_dll.create_ao_voltage_channel.argtypes = [
    C.c_void_p, C.c_char_p, C.c_char_p,
    C.c_double, C.c_double, C.c_int32, C.c_char_p,
]
_dll.create_ao_voltage_channel.restype = _check_error

_dll.create_do_channel = _dll.DAQmxCreateDOChan
_dll.create_do_channel.argtypes = [
    C.c_void_p, C.c_char_p, C.c_char_p, C.c_int32,
]
_dll.create_do_channel.restype = _check_error

_dll.clock_timing = _dll.DAQmxCfgSampClkTiming
_dll.clock_timing.argtypes = [
    C.c_void_p, C.c_char_p, C.c_double, C.c_int32, C.c_int32, C.c_uint64,
]
_dll.clock_timing.restype = _check_error

_dll.write_voltages = _dll.DAQmxWriteAnalogF64
_dll.write_voltages.argtypes = [
    C.c_void_p, C.c_int32, C.c_uint32, C.c_double, C.c_uint32,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),
    C.POINTER(C.c_int32), C.POINTER(C.c_uint32),
]
_dll.write_voltages.restype = _check_error

_dll.start_task = _dll.DAQmxStartTask
_dll.start_task.argtypes = [C.c_void_p]
_dll.start_task.restype = _check_error

_dll.finish_task = _dll.DAQmxWaitUntilTaskDone
_dll.finish_task.argtypes = [C.c_void_p, C.c_double]
_dll.finish_task.restype = _check_error

_dll.stop_task = _dll.DAQmxStopTask
_dll.stop_task.argtypes = [C.c_void_p]
_dll.stop_task.restype = _check_error

_dll.clear_task = _dll.DAQmxClearTask
_dll.clear_task.argtypes = [C.c_void_p]
_dll.clear_task.restype = _check_error


class DAQ:
    """Controller for a National Instruments PCIe-6363 analog output.

    Parameters
    ----------
    name : str
        A descriptive name for this device instance.
    num_channels : int
        Number of analog output channels to use (1-7).
    rate : float
        Sample rate in samples per second.
    board_name : str
        The device identifier as shown in NI MAX (e.g. 'Dev1').
    """

    MAX_RATE = 1e6

    def __init__(
        self,
        name: str = 'NI-DAQ-6363',
        num_channels: int = 7,
        rate: float = 1e4,
        board_name: str = 'Dev1',
    ):
        if not 1 <= num_channels <= 7:
            raise ValueError(
                f"num_channels must be between 1 and 7, got {num_channels}."
            )

        self.name = name
        self.num_channels = num_channels
        self.rate = rate

        logger.info("%s: opening...", self.name)

        self.device_name = bytes(
            f'{board_name}/ao0:{self.num_channels - 1}', 'ascii'
        )
        self.task_handle = C.c_void_p(0)
        self.num_points_written = C.c_int32(0)

        self._prepare_to_write_voltages()
        self._task_running = False
        self._task_loaded = False

        self.voltages = np.zeros((2, self.num_channels), 'float64')
        self.set_rate(self.rate)

        logger.info("%s: opened and ready.", self.name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def s2p(self, seconds: float) -> int:
        """Convert time in seconds to analog output sample count."""
        return int(round(self.rate * seconds))

    def p2s(self, num_pixels: int) -> float:
        """Convert analog output sample count to time in seconds."""
        return num_pixels / self.rate

    def s2s(self, seconds: float) -> float:
        """Round seconds to the nearest value deliverable by the AO clock."""
        return self.p2s(self.s2p(seconds))

    def set_rate(self, rate: float):
        """Set the sample clock rate.

        Parameters
        ----------
        rate : float
            Desired sample rate in Hz. Must be > 0 and <= MAX_RATE.
        """
        self._ensure_task_is_stopped()
        if not 0 < rate <= self.MAX_RATE:
            raise ValueError(
                f"Rate must be between 0 and {self.MAX_RATE}, got {rate}."
            )
        self.rate = float(rate)
        _dll.clock_timing(
            self.task_handle,
            None,
            self.rate,
            DAQMX_VAL_RISING,
            DAQMX_VAL_FINITE_SAMPS,
            self.voltages.shape[0],
        )

    def play_voltages(
        self,
        voltages: np.ndarray = None,
        force_final_zeros: bool = True,
        block: bool = True,
    ):
        """Play an array of voltages on the analog outputs.

        Parameters
        ----------
        voltages : np.ndarray or None
            2D array of shape (num_samples, num_channels). If None, replays
            the previously loaded voltages.
        force_final_zeros : bool
            If True, the last sample on all channels is forced to 0 V.
        block : bool
            If True, waits until the voltage play is finished before
            returning.
        """
        self._ensure_task_is_stopped()
        logger.info("%s: playing voltages...", self.name)
        if voltages is not None:
            self._write_voltages(voltages, force_final_zeros)
        _dll.start_task(self.task_handle)
        self._task_running = True
        self._task_loaded = False
        if block:
            self._ensure_task_is_stopped()

    def close(self):
        """Stop any running task and release hardware resources."""
        self._ensure_task_is_stopped()
        logger.info("%s: closing...", self.name)
        _dll.clear_task(self.task_handle)
        logger.info("%s: closed.", self.name)

    def _prepare_to_write_voltages(self):
        """Create the DAQmx task and configure analog output channels."""
        _dll.create_task(bytes(), self.task_handle)
        _dll.create_ao_voltage_channel(
            self.task_handle,
            self.device_name,
            b"",
            -10.0,
            +10.0,
            DAQMX_VAL_VOLTS,
            None,
        )

    def write_voltages(self, voltages: np.ndarray, force_final_zeros: bool = True):
        """Validate and write a voltage array to the DAQ buffer.

        Parameters
        ----------
        voltages : np.ndarray
            2D float64 array of shape (num_samples, num_channels).
        force_final_zeros : bool
            If True, the last sample on all channels is set to 0 V.
        """
        if voltages.ndim != 2:
            raise ValueError(
                f"voltages must be 2D, got {voltages.ndim}D."
            )
        if voltages.dtype != np.float64:
            raise TypeError(
                f"voltages must have dtype float64, got {voltages.dtype}."
            )
        if voltages.shape[0] < 2:
            raise ValueError(
                f"voltages must have at least 2 samples, got {voltages.shape[0]}."
            )
        if voltages.shape[1] != self.num_channels:
            raise ValueError(
                f"voltages must have {self.num_channels} channels, "
                f"got {voltages.shape[1]}."
            )

        logger.info("%s: writing voltages...", self.name)
        if force_final_zeros:
            logger.debug("%s: forcing final voltages to zero.", self.name)
            voltages[-1, :] = 0

        old_num_samples = self.voltages.shape[0]
        self.voltages = voltages
        self._ensure_task_is_stopped()

        if self._task_loaded:
            _dll.clear_task(self.task_handle)
            self._prepare_to_write_voltages()
            self.set_rate(self.rate)
        elif self.voltages.shape[0] != old_num_samples:
            self.set_rate(self.rate)

        _dll.write_voltages(
            self.task_handle,
            self.voltages.shape[0],
            0,
            10.0,
            DAQMX_VAL_GROUP_BY_SCAN_NUMBER,
            self.voltages,
            self.num_points_written,
            None,
        )
        logger.info(
            "%s: %i points written to each channel.",
            self.name, self.num_points_written.value,
        )
        self._task_loaded = True

    def _ensure_task_is_stopped(self):
        """Wait for a running task to finish, then stop it."""
        if self._task_running:
            logger.info("%s: waiting for voltage play to finish...", self.name)
            _dll.finish_task(self.task_handle, -1)
            logger.info("%s: done.", self.name)
            _dll.stop_task(self.task_handle)
            self._task_running = False


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    with DAQ(num_channels=3, rate=1e1) as ao:
        play_s = 100
        print('\nSine wave:')
        volts = np.zeros((ao.s2p(play_s), ao.num_channels), 'float64')
        volts[:, 0] = 1
        volts[:, 1] = 2
        volts[:, 2] = 3
        ao.play_voltages(volts)
