"""Thorlabs FW103 Filter Wheel Controller.

Controlled via a KCube Stepper Motor (KST201) using the Thorlabs
Kinesis .NET library through pythonnet (clr).
"""

import logging
import sys
import time
from typing import Dict, Optional

import clr

logger = logging.getLogger(__name__)

# Add Thorlabs Kinesis .NET assemblies
_KINESIS_PATH = r'C:\Program Files\Thorlabs\Kinesis'
sys.path.append(_KINESIS_PATH)

clr.AddReference(
    rf"{_KINESIS_PATH}\Thorlabs.MotionControl.DeviceManagerCLI.dll"
)
clr.AddReference(
    rf"{_KINESIS_PATH}\Thorlabs.MotionControl.GenericMotorCLI.dll"
)
clr.AddReference(
    rf"{_KINESIS_PATH}\ThorLabs.MotionControl.KCube.StepperMotorCLI.dll"
)

from Thorlabs.MotionControl.DeviceManagerCLI import (  # noqa: E402
    DeviceConfiguration,
    DeviceManagerCLI,
)
from Thorlabs.MotionControl.KCube.StepperMotorCLI import KCubeStepper  # noqa: E402
from System import Decimal  # noqa: E402


class FilterWheel:
    """Controller for the Thorlabs FW103 filter wheel.

    The FW103 has 6 positions, each separated by 60 degrees (360/6).
    It is driven by a KCube Stepper Motor (KST201).

    Parameters
    ----------
    serial_number : str
        Serial number of the KST201 controller (e.g. '26006458').
    filter_map : dict, optional
        Mapping of positions (1-6) to filter names. If None, a default
        map with empty slots is used.
    """

    DEGREES_PER_POSITION = 60.0
    NUM_POSITIONS = 6
    DEFAULT_TIMEOUT_MS = 60000  # 60 seconds

    def __init__(
        self,
        serial_number: str,
        filter_map: Optional[Dict[int, str]] = None,
    ):
        self.serial_number = serial_number
        self.device = None
        self._connected = False

        self.filter_map: Dict[int, str] = filter_map or {
            1: "Empty",
            2: "ET 605/52",
            3: "ET 525/50",
            4: "ET 510/20",
            5: "Empty",
            6: "Empty",
        }

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        """Connect to the KCube stepper and enable the device.

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
                f"Device {self.serial_number} not found. "
                f"Available devices: {devices}"
            )

        logger.info("Connecting to KST201 %s...", self.serial_number)
        self.device = KCubeStepper.CreateKCubeStepper(self.serial_number)
        self.device.Connect(self.serial_number)
        time.sleep(0.25)

        device_info = self.device.GetDeviceInfo()
        logger.info("Device: %s", device_info.Description)

        self.device.StartPolling(250)
        time.sleep(0.25)

        self.device.EnableDevice()
        time.sleep(0.25)

        use_file = DeviceConfiguration.DeviceSettingsUseOptionType.UseFileSettings
        self.device.LoadMotorConfiguration(self.device.DeviceID, use_file)

        self._connected = True
        logger.info("Device connected and enabled.")
        return True

    def disconnect(self):
        """Stop polling and disconnect from the device."""
        if self.device is not None:
            try:
                self.device.StopPolling()
                self.device.Disconnect()
                self._connected = False
                logger.info("Disconnected from %s.", self.serial_number)
            except Exception:
                logger.exception("Error while disconnecting.")

    def finalize(self):
        """Alias for disconnect(), for consistency with other controllers."""
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        """Whether the device is currently connected."""
        return self._connected

    # --- Position queries ---

    def get_position_degrees(self) -> float:
        """Get the current wheel position in degrees.

        Returns
        -------
        float
            Position in degrees (0-360).
        """
        if not self._connected:
            raise RuntimeError("Filter wheel is not connected.")
        pos_str = str(self.device.Position).replace(',', '.')
        return float(pos_str)

    def get_position(self) -> int:
        """Get the current filter slot number.

        Returns
        -------
        int
            Position number (1-6).
        """
        degrees = self.get_position_degrees()
        pos = int(round(degrees / self.DEGREES_PER_POSITION)) % self.NUM_POSITIONS
        if pos == 0:
            pos = self.NUM_POSITIONS
        return pos

    def get_current_filter(self) -> str:
        """Get the name of the filter in the current position.

        Returns
        -------
        str
            Filter name from the filter map.
        """
        pos = self.get_position()
        return self.filter_map.get(pos, "Unknown")

    # --- Motion control ---

    def move_to_position(
        self,
        position: int,
        timeout_ms: int = None,
    ) -> bool:
        """Move the wheel to a specific slot.

        Parameters
        ----------
        position : int
            Target position (1-6).
        timeout_ms : int, optional
            Movement timeout in milliseconds. Defaults to
            DEFAULT_TIMEOUT_MS.

        Returns
        -------
        bool
            True if the final position matches the target.
        """
        if not self._connected:
            raise RuntimeError("Filter wheel is not connected.")
        if not 1 <= position <= self.NUM_POSITIONS:
            raise ValueError(
                f"Position must be between 1 and {self.NUM_POSITIONS}, "
                f"got {position}."
            )
        if timeout_ms is None:
            timeout_ms = self.DEFAULT_TIMEOUT_MS

        target_degrees = position * self.DEGREES_PER_POSITION
        logger.info(
            "Moving to position %i (%.1f degrees)...",
            position, target_degrees,
        )
        self.device.MoveTo(Decimal(target_degrees), timeout_ms)
        time.sleep(0.5)  # stabilization pause

        final_pos = self.get_position()
        logger.info(
            "Final position: %i (%.2f degrees).",
            final_pos, self.get_position_degrees(),
        )
        return final_pos == position

    def move_to_filter(self, filter_name: str, timeout_ms: int = None) -> bool:
        """Move the wheel to a named filter.

        Parameters
        ----------
        filter_name : str
            Name of the filter (e.g. 'ET 510/20'). Must match an entry
            in the filter map.
        timeout_ms : int, optional
            Movement timeout in milliseconds.

        Returns
        -------
        bool
            True if the final position matches the target.

        Raises
        ------
        ValueError
            If the filter name is not found in the filter map.
        """
        for pos, name in self.filter_map.items():
            if name == filter_name:
                return self.move_to_position(pos, timeout_ms)
        raise ValueError(
            f"Filter '{filter_name}' not found in the filter map. "
            f"Available filters: {self.get_filter_names()}"
        )

    def home(self) -> bool:
        """Move the wheel to position 1.

        Returns
        -------
        bool
            True if homing was successful.
        """
        return self.move_to_position(1)

    # --- Filter map management ---

    def set_filter_map(self, filter_map: Dict[int, str]):
        """Replace the position-to-filter-name mapping.

        Parameters
        ----------
        filter_map : dict
            Mapping of position numbers (1-6) to filter names.
        """
        self.filter_map = filter_map

    def get_filter_names(self) -> list:
        """Get the names of installed filters (excluding 'Empty' slots).

        Returns
        -------
        list of str
            Names of installed filters.
        """
        return [name for name in self.filter_map.values() if name != "Empty"]

    def get_all_positions(self) -> Dict[int, str]:
        """Get a copy of the full position-to-filter mapping.

        Returns
        -------
        dict
            Mapping of position numbers to filter names.
        """
        return self.filter_map.copy()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    SERIAL_NUMBER = "26006458"

    with FilterWheel(SERIAL_NUMBER) as fw:
        print(
            f"Position: {fw.get_position()} - "
            f"Filter: {fw.get_current_filter()}"
        )
        print("Moving to position 6...")
        fw.move_to_position(6)
        print(
            f"Position: {fw.get_position()} - "
            f"Filter: {fw.get_current_filter()}"
        )
