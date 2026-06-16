"""Oxxius LBX 488 Laser Controller.

Reference manual:
https://www.optoprim.it/wp-content/uploads/2020/03/LBX-series-Manual.pdf
"""

import logging
from time import sleep

import serial

logger = logging.getLogger(__name__)


class OxxiusLaser488:
    """Controller for the Oxxius LBX 488 laser via serial communication.

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
        'write_termination': '\n',
        'read_termination': '\r\n',
        'encoding': 'ascii',
        'decoding': 'utf-8',
        'baudrate': 19200,
        'read_timeout': 1,
        'write_timeout': 1,
        'bytes_to_read': 200,
    }

    FAULT_CODES = {
        '0': 'No alarm',
        '1': 'Diode current',
        '2': 'Laser power',
        '3': 'Power supply',
        '4': 'Diode temperature',
        '5': 'Base temperature',
        '6': 'Warning end of life',
    }

    STATUS_CODES = {
        '1': 'Warming up',
        '2': 'Standby',
        '3': 'Emission on',
        '5': 'Alarm present',
        '6': 'Sleep',
        '7': 'Searching for SLM point',
    }

    ANALOG_CONTROL_MODES = {
        '0': 'Power',
        '1': 'Current',
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
        """Open the serial port with the default settings."""
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

    def finalize(self):
        """Close the serial resource."""
        if self.rsc is not None:
            self.rsc.close()
            sleep(0.2)

    def query(self, message: str) -> str:
        """Send a command to the device and return its response.

        Parameters
        ----------
        message : str
            The command to send to the device.

        Returns
        -------
        str
            The device response.
        """
        encoded = (message + self.DEFAULTS['write_termination']).encode(
            self.DEFAULTS['encoding']
        )
        self.rsc.write(encoded)

        timeout = self.DEFAULTS['read_timeout']
        elapsed = 0.0
        poll_interval = 0.01
        while self.rsc.in_waiting == 0:
            sleep(poll_interval)
            elapsed += poll_interval
            if elapsed >= timeout:
                raise TimeoutError(
                    f"No response from device after {timeout}s "
                    f"for command: {message}"
                )

        ans = self.rsc.read(self.DEFAULTS['bytes_to_read'])
        ans = ans.decode(self.DEFAULTS['decoding'])
        ans = ans.rstrip(self.DEFAULTS['read_termination'])
        return ans

    # --- Identification ---

    def idn(self) -> str:
        """Get the serial number from the device.

        Returns
        -------
        str
            The serial number and wavelength of the device.
        """
        return self.query('?HID')

    def ask_model(self) -> str:
        """Get the laser model identifier.

        Returns
        -------
        str
            e.g. 'LBX-488-50' for a 50 mW LBX emitting at 488 nm.
        """
        return self.query('?INF')

    # --- Configuration and status ---

    def ask_analog_control_mode(self) -> str:
        """Get the analog control mode (power or current).

        Returns
        -------
        str
            'Power' or 'Current'.
        """
        reply = self.query('?ACC')
        return self.ANALOG_CONTROL_MODES.get(
            reply, f'Unknown analog control mode: {reply}'
        )

    def ask_base_temp(self) -> str:
        """Get the measured base plate temperature.

        Returns
        -------
        str
            Temperature in degrees Celsius.
        """
        return self.query('?BT')

    def ask_diode_temp(self) -> str:
        """Get the measured laser diode temperature.

        Returns
        -------
        str
            Temperature in degrees Celsius.
        """
        return self.query('?DT')

    def ask_processor_temp(self) -> str:
        """Get the microcontroller temperature inside the laser head.

        Returns
        -------
        str
            Temperature in degrees Celsius.
        """
        return self.query('?PST')

    def ask_current(self) -> str:
        """Get the measured laser diode current.

        Returns
        -------
        str
            Current in mA.
        """
        return self.query('?C')

    def fault(self) -> str:
        """Get the cause of the latest alarm.

        Returns
        -------
        str
            Description of the fault.
        """
        code = self.query('?F')
        return self.FAULT_CODES.get(code, f'Unknown fault code: {code}')

    def ask_hours(self) -> str:
        """Get the accumulated operating hours.

        Returns
        -------
        str
            Operating hours.
        """
        return self.query('?HH')

    def ask_power(self) -> str:
        """Get the measured output power.

        Returns
        -------
        str
            Output power in mW.
        """
        return self.query('?P')

    def ask_status(self) -> str:
        """Get the current laser status.

        Returns
        -------
        str
            Description of the laser status.
        """
        code = self.query('?STA')
        return self.STATUS_CODES.get(code, f'Unknown status code: {code}')

    # --- Power and operational settings ---

    def set_laser_mode(self, action: str) -> str:
        """Set the analog control mode.

        Parameters
        ----------
        action : str
            'Power' or 'Current'.

        Returns
        -------
        str
            Device response.
        """
        commands = {'Power': 'ACC 0', 'Current': 'ACC 1'}
        if action not in commands:
            raise ValueError(
                f"Invalid action '{action}'. Must be 'Power' or 'Current'."
            )
        return self.query(commands[action])

    def set_laser_control(self, action: str) -> str:
        """Turn the laser on or off.

        Parameters
        ----------
        action : str
            'ON' or 'OFF'.

        Returns
        -------
        str
            Device response.
        """
        commands = {'OFF': 'L 0', 'ON': 'L 1'}
        if action not in commands:
            raise ValueError(f"Invalid action '{action}'. Must be 'ON' or 'OFF'.")
        logger.info("Laser 488 %s", action)
        return self.query(commands[action])

    def set_diode_current(self, current: float) -> str:
        """Set the laser diode current.

        Parameters
        ----------
        current : float
            Desired current as a percentage (0-125).

        Returns
        -------
        str
            Device response.
        """
        if current >= 125:
            raise ValueError(
                f"Current {current}% out of range. Must be 0-125%."
            )
        return self.query(f'C {current}')

    def set_laser_power(self, power: float) -> str:
        """Set the laser output power.

        Parameters
        ----------
        power : float
            Desired power in mW. Must be less than the maximum power.

        Returns
        -------
        str
            Device response.
        """
        max_power = float(self.query('?MAXLP'))
        if power >= max_power:
            raise ValueError(
                f"Power {power} mW exceeds maximum ({max_power} mW). "
                f"Range: 0 to {max_power} mW."
            )
        return self.query(f'P {power}')

    def set_shutter(self, action: str) -> str:
        """Simulate shutter control by toggling laser emission.

        Note: This laser has no physical shutter. 'OPEN' turns emission on
        and 'CLOSE' turns emission off.

        Parameters
        ----------
        action : str
            'OPEN' or 'CLOSE'.

        Returns
        -------
        str
            Device response.
        """
        commands = {'CLOSE': 'L 0', 'OPEN': 'L 1'}
        if action not in commands:
            raise ValueError(
                f"Invalid action '{action}'. Must be 'OPEN' or 'CLOSE'."
            )
        logger.info("Laser 488 shutter %s", action.lower())
        return self.query(commands[action])


if __name__ == '__main__':
    with OxxiusLaser488('COM7') as dev:  # <-- Remember to change the port
        serial_number = dev.idn()
        print(f'The device serial number is: {serial_number}')
