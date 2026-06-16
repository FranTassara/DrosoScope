"""Thorlabs MFF101 Motorized Filter Flipper Controller.

Controlled via the Thorlabs Kinesis .NET library through pythonnet (clr).

Positions
---------
Position 1 : mirror in "up" state
Position 2 : mirror in "down" state
"""

import logging
import sys
import time

import clr

logger = logging.getLogger(__name__)

_KINESIS_PATH = r'C:\Program Files\Thorlabs\Kinesis'
sys.path.append(_KINESIS_PATH)

clr.AddReference(
    rf"{_KINESIS_PATH}\Thorlabs.MotionControl.DeviceManagerCLI.dll"
)
clr.AddReference(
    rf"{_KINESIS_PATH}\Thorlabs.MotionControl.FilterFlipperCLI.dll"
)

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI      # noqa: E402
from Thorlabs.MotionControl.FilterFlipperCLI import FilterFlipper         # noqa: E402


class MFF101:
    """Controller for the Thorlabs MFF101 motorized filter flipper.

    Parameters
    ----------
    serial_number : str
        Serial number of the MFF101 (e.g. '37009524').
    """

    POSITION_1 = 1
    POSITION_2 = 2
    DEFAULT_TIMEOUT_MS = 10000  # 10 seconds

    def __init__(self, serial_number: str):
        self.serial_number = serial_number
        self.device = None
        self._connected = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        """Build device list, connect and enable the flipper.

        Raises
        ------
        RuntimeError
            If the device is not found or connection fails.
        """
        DeviceManagerCLI.BuildDeviceList()
        device_list = DeviceManagerCLI.GetDeviceList()
        devices = [str(device_list[i]) for i in range(device_list.Count)]
        logger.debug("Devices found: %s", devices)

        if self.serial_number not in devices:
            raise RuntimeError(
                f"MFF101 {self.serial_number} not found. "
                f"Available devices: {devices}"
            )

        logger.info("Connecting to MFF101 %s...", self.serial_number)
        self.device = FilterFlipper.CreateFilterFlipper(self.serial_number)
        self.device.Connect(self.serial_number)
        time.sleep(0.25)

        self.device.StartPolling(250)
        time.sleep(0.25)

        self.device.EnableDevice()
        time.sleep(0.25)

        self._connected = True
        logger.info("MFF101 %s connected.", self.serial_number)
        return True

    def disconnect(self):
        """Stop polling and disconnect."""
        if self.device is not None:
            try:
                self.device.StopPolling()
                self.device.Disconnect()
                self._connected = False
                logger.info("MFF101 %s disconnected.", self.serial_number)
            except Exception:
                logger.exception("Error disconnecting MFF101 %s.", self.serial_number)

    def finalize(self):
        """Alias for disconnect(), consistent with other controllers."""
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Position control
    # ------------------------------------------------------------------

    def set_position(self, position: int, timeout_ms: int = None):
        """Move the flipper to position 1 or 2.

        Parameters
        ----------
        position : int
            Target position (1 or 2).
        timeout_ms : int, optional
            Timeout in milliseconds. Defaults to DEFAULT_TIMEOUT_MS.

        Raises
        ------
        RuntimeError
            If the flipper is not connected.
        ValueError
            If position is not 1 or 2.
        """
        if not self._connected:
            raise RuntimeError(f"MFF101 {self.serial_number} is not connected.")
        if position not in (1, 2):
            raise ValueError(f"Position must be 1 or 2, got {position}.")

        if timeout_ms is None:
            timeout_ms = self.DEFAULT_TIMEOUT_MS

        logger.info("MFF101 %s → position %i", self.serial_number, position)
        self.device.SetPosition(position, timeout_ms)

    def get_position(self) -> int:
        """Return the current position (1 or 2).

        Returns
        -------
        int
        """
        if not self._connected:
            raise RuntimeError(f"MFF101 {self.serial_number} is not connected.")
        return int(self.device.Position)
