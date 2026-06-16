"""ASI MS-2000 Motorized Stage Controller.

Reference:
https://asiimaging.com/docs/products/serial_commands
"""

import logging
from time import sleep

import serial

logger = logging.getLogger(__name__)


class AsiStage:
    """Controller for the ASI MS-2000 stage via serial communication.

    Parameters
    ----------
    port : str
        The port where the device is connected.
        e.g. 'COM3' on Windows or '/dev/ttyACM0' on Linux.

    Attributes
    ----------
    rsc : serial.Serial or None
        The serial communication resource.
    port : str
        The port where the device is connected.
    """

    DEFAULTS = {
        'write_termination': '\r',
        'read_termination': '\r\n',
        'encoding': 'ascii',
        'decoding': 'ascii',
        'baudrate': 9600,
        'read_timeout': 1,
        'write_timeout': 1,
    }

    def __init__(self, port: str):
        self.port = port
        self.rsc = None

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()

    def initialize(self):
        """Open the serial port and configure buffer sizes."""
        self.rsc = serial.Serial(
            port=self.port,
            baudrate=self.DEFAULTS['baudrate'],
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.DEFAULTS['read_timeout'],
            write_timeout=self.DEFAULTS['write_timeout'],
        )
        sleep(1)
        self.rsc.close()
        self.rsc.set_buffer_size(12800, 12800)

        try:
            self.rsc.open()
            logger.info("ASI stage connected on %s.", self.port)
        except serial.SerialException:
            logger.error("Failed to connect to ASI stage on %s.", self.port)
            raise

    def finalize(self):
        """Close the serial resource."""
        if self.rsc is not None:
            self.rsc.close()
            sleep(0.2)

    def query(self, message: str) -> str:
        """Send a command to the stage and return its response.

        Parameters
        ----------
        message : str
            The command to send (without termination character).

        Returns
        -------
        str
            The device response.
        """
        self.rsc.reset_input_buffer()
        self.rsc.reset_output_buffer()
        encoded = bytes(
            f"{message}{self.DEFAULTS['write_termination']}",
            encoding=self.DEFAULTS['encoding'],
        )
        self.rsc.write(encoded)
        ans = self.rsc.readline()
        return ans.decode(encoding=self.DEFAULTS['decoding'])

    # --- Identification ---

    def idn(self) -> str:
        """Get the device identification string.

        Returns
        -------
        str
            The device serial number / identification.
        """
        return self.query('WHO')

    # --- Motion control ---

    def halt(self) -> str:
        """Immediately stop all axes.

        Returns
        -------
        str
            Device response.
        """
        return self.query('HALT')

    def zero(self) -> str:
        """Set the current position as the origin for all axes.

        Returns
        -------
        str
            Device response.
        """
        return self.query('ZERO')

    def home(self) -> str:
        """Move all axes to their home positions.

        Returns
        -------
        str
            Device response.
        """
        return self.query('HOME')

    def move_relative(self, x: int = 0, y: int = 0, z: int = 0) -> str:
        """Move all axes by a relative distance.

        Parameters
        ----------
        x, y, z : int
            Relative displacement in ASI units (tenths of microns).

        Returns
        -------
        str
            Device response.
        """
        return self.query(f'MOVREL X={x} Y={y} Z={z}')

    def move_relative_axis(self, axis: str, distance: int = 0) -> str:
        """Move a single axis by a relative distance.

        Parameters
        ----------
        axis : str
            Axis name ('X', 'Y', or 'Z').
        distance : int
            Relative displacement in ASI units (tenths of microns).

        Returns
        -------
        str
            Device response.
        """
        return self.query(f'MOVREL {axis}={distance}')

    def move_absolute(self, x: int = 0, y: int = 0, z: int = 0) -> str:
        """Move all axes to absolute positions.

        Parameters
        ----------
        x, y, z : int
            Target positions in ASI units (tenths of microns).

        Returns
        -------
        str
            Device response.
        """
        return self.query(f'MOVE X={x} Y={y} Z={z}')

    def move_absolute_axis(self, axis: str, position: int = 0) -> str:
        """Move a single axis to an absolute position.

        Parameters
        ----------
        axis : str
            Axis name ('X', 'Y', or 'Z').
        position : int
            Target position in ASI units (tenths of microns).

        Returns
        -------
        str
            Device response.
        """
        return self.query(f'MOVE {axis}={position}')

    # --- Speed and acceleration ---

    def set_max_speed(self, axis: str, speed: float) -> str:
        """Set the maximum speed for a specific axis.

        Parameters
        ----------
        axis : str
            Axis name ('X', 'Y', or 'Z').
        speed : float
            Speed in mm/s.

        Returns
        -------
        str
            Device response.
        """
        return self.query(f'SPEED {axis}={speed}')

    def set_accel(self, axis: str, accel: int) -> str:
        """Set the acceleration ramp time for an axis.

        Parameters
        ----------
        axis : str
            Axis name ('X', 'Y', or 'Z').
        accel : int
            Time in milliseconds from stopped to maximum speed.

        Returns
        -------
        str
            Device response.
        """
        return self.query(f'ACCEL {axis}={accel}')

    # --- Position queries ---

    def ask_position(self, axis: str) -> int:
        """Get the current position of an axis in ASI units.

        Parameters
        ----------
        axis : str
            Axis name ('X', 'Y', or 'Z').

        Returns
        -------
        int
            Position in ASI units (tenths of microns).
        """
        sleep(0.2)
        reply = self.query(f'WHERE {axis}')
        return int(reply.split(" ")[1])

    def ask_position_um(self, axis: str) -> float:
        """Get the current position of an axis in microns.

        Parameters
        ----------
        axis : str
            Axis name ('X', 'Y', or 'Z').

        Returns
        -------
        float
            Position in microns.
        """
        sleep(0.2)
        reply = self.query(f'WHERE {axis}')
        return float(reply.split(" ")[1]) / 10.0

    # --- Status ---

    def is_axis_busy(self, axis: str) -> bool:
        """Check whether a specific axis is currently moving.

        Parameters
        ----------
        axis : str
            Axis name ('X', 'Y', or 'Z').

        Returns
        -------
        bool
            True if the axis is busy.
        """
        reply = self.query(f'RS {axis}?')
        return 'B' in reply

    def is_device_busy(self) -> bool:
        """Check whether any axis is currently moving.

        Returns
        -------
        bool
            True if any axis is busy.
        """
        reply = self.query('/')
        return 'B' in reply

    def wait_for_device(self) -> None:
        """Block until all axes have stopped moving."""
        logger.info("%s: waiting for device...", self.port)
        while self.is_device_busy():
            pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    with AsiStage('COM7') as dev:  # <-- Remember to change the port
        print(dev.idn())
        print('Moving the stage')
        dev.move_relative_axis('X', 1)
        dev.wait_for_device()
        print(dev.ask_position('X'))
        dev.zero()
        print('Finished')
