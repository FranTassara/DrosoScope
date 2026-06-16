"""Oxxius LCX 561 Laser Controller."""

import logging
from time import sleep

import serial

logger = logging.getLogger(__name__)


class OxxiusLaser561:
    """Controller for the Oxxius LCX 561 laser via serial communication.

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
        '7': 'Interlock',
        '8': 'User-generated alarm (using the command RST)',
    }

    STATUS_CODES = {
        '1': 'Warming up',
        '2': 'Standby',
        '3': 'Emission on',
        '5': 'Alarm present',
        '6': 'Sleep',
        '7': 'Searching for SLM point',
    }

    EMISSION_CODES = {
        '0': 'Emission is off',
        '1': 'Emission is on',
        '2': 'Emission is on at low power',
    }

    INTERLOCK_CODES = {
        '0': 'Interlock open, laser emission is not authorized',
        '1': 'Interlock closed, laser emission is authorized',
    }

    CDRH_CODES = {
        '0': 'No delay between emission command and actual emission',
        '1': 'A five-second delay is enforced between emission command and actual emission',
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
            e.g. 'LAS-xxxxx' where xxxxx is a five-digit number.
        """
        return self.query('?HID')

    def ask_model(self) -> str:
        """Get the laser model identifier.

        Returns
        -------
        str
            e.g. 'LCX-532-50' for a 50 mW LCX emitting at 532 nm.
        """
        return self.query('?INF')

    def ask_version(self) -> str:
        """Get the firmware version.

        Returns
        -------
        str
            e.g. '1.6.8'
        """
        return self.query('?SV')

    # --- Configuration and status ---

    def ask_base_temp(self) -> str:
        """Get the measured base plate temperature.

        Returns
        -------
        str
            Temperature in degrees Celsius.
        """
        return self.query('?BT')

    def ask_cdrh(self) -> str:
        """Get the CDRH delay status.

        Returns
        -------
        str
            Description of the current CDRH delay setting.
        """
        reply = self.query('?CDRH')
        return self.CDRH_CODES.get(reply, f'Unknown CDRH code: {reply}')

    def set_cdrh(self, action: str) -> str:
        """Enable or disable the CDRH emission delay.

        Parameters
        ----------
        action : str
            'ON' to enable, 'OFF' to disable.

        Returns
        -------
        str
            Device response.
        """
        commands = {'OFF': 'CDRH 0', 'ON': 'CDRH 1'}
        if action not in commands:
            raise ValueError(f"Invalid action '{action}'. Must be 'ON' or 'OFF'.")
        logger.info("Laser 561 CDRH %s", action)
        return self.query(commands[action])

    def fault(self) -> str:
        """Get the cause of the latest alarm.

        Returns
        -------
        str
            Description of the fault.
        """
        code = self.query('?F')
        return self.FAULT_CODES.get(code, f'Unknown fault code: {code}')

    def ask_interlock(self) -> str:
        """Get the interlock circuit status.

        Returns
        -------
        str
            Description of the interlock state.
        """
        reply = self.query('?INT')
        return self.INTERLOCK_CODES.get(reply, f'Unknown interlock code: {reply}')

    def ask_voltage(self) -> str:
        """Get the voltage supplying the laser head.

        Returns
        -------
        str
            Voltage in Volts.
        """
        return self.query('?IV')

    def ask_hours(self) -> str:
        """Get the accumulated operating hours.

        Returns
        -------
        str
            Operating hours.
        """
        return self.query('?HH')

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

    def ask_power(self) -> str:
        """Get the measured output power.

        Returns
        -------
        str
            Output power in mW.
        """
        return self.query('?P')

    def ask_emission(self) -> str:
        """Get the emission status.

        Returns
        -------
        str
            Description of the emission state.
        """
        reply = self.query('?L')
        return self.EMISSION_CODES.get(reply, f'Unknown emission code: {reply}')

    def set_laser_control(self, action: str) -> str:
        """Turn the laser on, off, or set to low power.

        Parameters
        ----------
        action : str
            'ON', 'OFF', or 'LOW'.

        Returns
        -------
        str
            Device response.
        """
        commands = {'OFF': 'L 0', 'ON': 'L 1', 'LOW': 'L 2'}
        if action not in commands:
            raise ValueError(
                f"Invalid action '{action}'. Must be 'ON', 'OFF', or 'LOW'."
            )
        logger.info("Laser 561 %s", action)
        return self.query(commands[action])

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
        max_power = int(self.query('?MAXLP'))
        if power >= max_power:
            raise ValueError(
                f"Power {power} mW exceeds maximum ({max_power} mW). "
                f"Range: 0 to {max_power} mW."
            )
        return self.query(f'P {power}')

    def set_shutter(self, action: str) -> str:
        """Open or close the laser shutter.

        Parameters
        ----------
        action : str
            'OPEN' or 'CLOSE'.

        Returns
        -------
        str
            Device response.
        """
        commands = {'CLOSE': 'SH 0', 'OPEN': 'SH 1'}
        if action not in commands:
            raise ValueError(
                f"Invalid action '{action}'. Must be 'OPEN' or 'CLOSE'."
            )
        logger.info("Laser 561 shutter %s", action.lower())
        return self.query(commands[action])


if __name__ == '__main__':
    with OxxiusLaser561('COM7') as dev:  # <-- Remember to change the port
        serial_number = dev.idn()
        print(f'The device serial number is: {serial_number}')
