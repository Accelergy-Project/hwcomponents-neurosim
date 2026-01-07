from abc import ABC
import logging
import math
import sys
import os
from typing import Dict
from textwrap import dedent
from hwcomponents import ComponentModel, action
import hwcomponents_neurosim.neurointerface as neurointerface

# ==================================================================================================
# Constants
# ==================================================================================================
SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
DEBUG = False
PRIORITY = 0.3
PERMITTED_TECH_NODES = [7e-9, 10e-9, 14e-9, 22e-9, 32e-9, 45e-9, 65e-9, 90e-9, 130e-9]
SAMPLE_CELLS = [
    f.split(".")[0]
    for f in os.listdir(os.path.join(SCRIPT_DIR, "cells"))
    if f.endswith(".cell")
]

CACHE = {}

# Format:
#   Key: (Docstring, Default value)
#   If REQUIRED in docstring, then a value is required. Otherwise, default is populated.
#   If a parameter is not used in an estimation (e.g. creating a shift+add), then the defaults will
#   be used for the non-selected device (PIM_PARAMS defaults for rows & columns are used).
SHARED_PARAMS = {
    "cycle_period": (
        f"REQUIRED: Duration of one cycle in seconds",
        1e-9,
        float,
    ),
    "tech_node": (
        f"REQUIRED: tech_node node. Must be between {max(PERMITTED_TECH_NODES)}  "
        f"and {min(PERMITTED_TECH_NODES)}.",
        32e-9,
        float,
    ),
}


PIM_PARAMS = {
    **{
        "rows": ("REQUIRED: Number of rows in a crossbar.", 32),
        "cols": ("REQUIRED: Number of columns in a crossbar.", 32),
        "cols_active_at_once": ("REQUIRED: Number of columns active at once.", 8),
        "cell_config": (
            f"REQUIRED: Path to a NVSim cell config file to use, or one of "
            f"the following samples: " + f", ".join(f'"{s}"' for s in SAMPLE_CELLS),
            "placeholder",
            str,
        ),
        "average_input_value": (
            f" REQUIRED: Average input value to a row. Must be between 0 and 1.",
            1,
            float,
        ),
        "average_cell_value": (
            f" REQUIRED: Average cell value. Must be between 0 and 1.",
            1,
            float,
        ),
        "sequential": (
            f"OPTIONAL: Sequential mode. Default is False. If True, the "
            f"crossbar will be set up to activate one row at a time. "
            f"Can be used as a memory this way.",
            False,
        ),
        "adc_resolution": (
            f"OPTIONAL: ADC resolution. Set this if to use Neurosim's "
            f"build-in ADC. Default is False.",
            0,
        ),
        "read_pulse_width": (
            f"OPTIONAL: Read pulse width. Default is 10ns.",
            1e-8,
            float,
        ),
        "voltage_dac_bits": (
            f"OPTIONAL: Resolution of a voltage DAC for inputs.",
            1,
            int,
        ),
        "temporal_dac_bits": (
            f"OPTIONAL: Resolution of a temporal DAC for inputs.",
            1,
            int,
        ),
        "temporal_spiking": (
            f"OPTIONAL: Whether to use a spiking (#pulses) or a PWM (pulse  "
            f"length) approach for temporal DAC. Default is True ",
            False,
            bool,
        ),
        "voltage": (
            f"OPTIONAL: Supply voltage. Default set by the tech_node node.",
            0,
            float,
        ),
        "threshold_voltage": (
            f"OPTIONAL: Threshold voltage. Default set by the tech_node node.",
            0,
            float,
        ),
    },
    **SHARED_PARAMS,
}

ADDER_PARAMS = {
    **{
        "n_bits": (f"REQUIRED: # Bits of the adder.", 8),
    },
    **SHARED_PARAMS,
}

SHIFT_ADD_PARAMS = {
    **{
        "n_bits": (f"REQUIRED: # Bits of the adder.", 8),
        "shift_register_n_bits": (f"REQUIRED: # Bits of the shift register.", 16),
    },
    **SHARED_PARAMS,
}
MAX_POOL_PARAMS = {
    **{
        "n_bits": (f"REQUIRED: # Bits.", 8),
        "pool_window": (f"REQUIRED: Window size of max pooling.", 2),
    },
    **SHARED_PARAMS,
}
ADDER_TREE_PARAMS = {
    **{
        "n_bits": (f"REQUIRED: # Bits of the leaf adder.", 8),
        "n_adder_tree_inputs": (f"REQUIRED: Number of inputs to the adder tree.", 2),
    },
    **SHARED_PARAMS,
}
MUX_PARAMS = {
    **{
        "n_mux_inputs": (f"REQUIRED: Number of inputs to the mux.", 2),
        "n_bits": (f"REQUIRED: # Bits of the mux.", 8),
    },
    **SHARED_PARAMS,
}
FLIP_FLOP_PARAMS = {
    **{
        "n_bits": (f"REQUIRED: # Bits of flip-flop.", 8),
    },
    **SHARED_PARAMS,
}

PARAM_DICTS = [
    PIM_PARAMS,
    ADDER_PARAMS,
    SHIFT_ADD_PARAMS,
    MAX_POOL_PARAMS,
    ADDER_TREE_PARAMS,
    MUX_PARAMS,
    FLIP_FLOP_PARAMS,
]

PERIPHERAL_PARAMS = {
    **ADDER_PARAMS,
    **SHIFT_ADD_PARAMS,
    **MAX_POOL_PARAMS,
    **ADDER_TREE_PARAMS,
    **MUX_PARAMS,
    **FLIP_FLOP_PARAMS,
}
ALL_PARAMS = {
    **PIM_PARAMS,
    **ADDER_PARAMS,
    **SHIFT_ADD_PARAMS,
    **MAX_POOL_PARAMS,
    **ADDER_TREE_PARAMS,
    **MUX_PARAMS,
    **FLIP_FLOP_PARAMS,
}

SUPPORTED_CLASSES = {
    "array_row_drivers": (neurointerface.row_stats, PIM_PARAMS),
    "array_col_drivers": (neurointerface.col_stats, PIM_PARAMS),
    "array_adc": (neurointerface.col_stats, PIM_PARAMS),
    "memory_cell": (neurointerface.cell_stats, PIM_PARAMS),
    "shift_add": (neurointerface.shift_add_stats, SHIFT_ADD_PARAMS),
    "intadder": (neurointerface.adder_stats, ADDER_PARAMS),
    "intadder_tree": (neurointerface.adder_tree_stats, ADDER_TREE_PARAMS),
    "max_pool": (neurointerface.max_pool_stats, MAX_POOL_PARAMS),
    "mux": (neurointerface.mux_stats, MUX_PARAMS),
    "flip_flop": (neurointerface.flip_flop_stats, FLIP_FLOP_PARAMS),
    "not_gate": (neurointerface.not_gate_stats, SHARED_PARAMS),
    "nand_gate": (neurointerface.nand_gate_stats, SHARED_PARAMS),
    "nor_gate": (neurointerface.nor_gate_stats, SHARED_PARAMS),
}


class _NeurosimPlugInComponent(ComponentModel):
    """
    A base class for Neurosim plug-in components.

    Parameters
    ----------
    cell_config : str
        The path to the  cell config file to use.
    tech_node : str
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    rows : int
        The number of rows in the CiM array.
    cols : int
        The number of columns in the CiM array.
    cols_active_at_once : int
        The number of columns active at once. If this is less than the number of
        columns, the array is activated in multiple steps.
    adc_resolution : int, optional
        The resolution of the ADC. Resolution of 0 means no ADC, and the CiM array
        outputs leave as analog signals.
    read_pulse_width : float, optional
        The width of the read pulse in seconds. This is the duration for which the row
        is asserted when reading from the CiM array.
    voltage_dac_bits : int, optional
        The number of bits of a voltage DAC, which outputs analog signals by setting
        voltage to a given value.
    temporal_dac_bits : int, optional
        The number of bits of a temporal DAC, which outputs analog signals by setting
        the pulse width of a signal. Read pulse width will be multiplied by
        (2^temporal_dac_bits - 1).
    temporal_spiking : bool, optional
        Whether to use a spiking (#pulses) or a PWM (pulse length) approach for temporal
        DAC.
    voltage : float, optional
        The supply voltage in volts.
    threshold_voltage : float, optional
        The threshold voltage of all transistors, in volts.
    sequential : bool, optional
        Whether to use a sequential approach for the CiM array, meaing that rows are
        activated one at a time.
    average_input_value : float, optional
        The average input value to a row. Must be between 0 and 1.
    average_cell_value : float, optional
        The average value of a cell. Must be between 0 and 1.
    n_bits : int, optional
        The number of bits of digital components such as adders.
    shift_register_n_bits : int, optional
        The number of bits of shift registers.
    pool_window : int, optional
        Used only in max pool units. The window size of max pooling.
    n_adder_tree_inputs : int, optional
        Used only in adder trees. The number of inputs to the adder tree.
    n_mux_inputs : int, optional
        Used only in muxes. The number of inputs to the mux.
    """

    component_name = "_override_this_name_"
    priority = PRIORITY
    _get_stats_func = None  # To be overridden by child classes
    _params = []  # To be overridden by child classes

    def __init__(
        self,
        cell_config: str,
        tech_node: float,
        cycle_period: float = 100e-9,
        rows: int = 32,
        cols: int = 32,
        cols_active_at_once: int = 8,
        adc_resolution: int = 0,
        read_pulse_width: float = 1e-8,
        voltage_dac_bits: int = 1,
        temporal_dac_bits: int = 1,
        temporal_spiking: bool = True,
        voltage: float = 0,
        threshold_voltage: float = 0,
        sequential: bool = False,
        average_input_value: float = 1,
        average_cell_value: float = 1,
        n_bits: int = 8,
        shift_register_n_bits: int = 16,
        pool_window: int = 2,
        n_adder_tree_inputs: int = 2,
        n_mux_inputs: int = 2,
    ):
        self.cell_config = cell_config
        self.tech_node = tech_node
        self.cycle_period = cycle_period
        self.rows = rows
        self.cols = cols
        self.cols_active_at_once = cols_active_at_once
        self.adc_resolution = adc_resolution
        self.read_pulse_width = read_pulse_width
        self.voltage_dac_bits = voltage_dac_bits
        self.temporal_dac_bits = temporal_dac_bits
        self.temporal_spiking = temporal_spiking
        self.voltage = voltage
        self.threshold_voltage = threshold_voltage
        self.sequential = sequential
        self.average_input_value = average_input_value
        self.average_cell_value = average_cell_value
        self.n_bits = n_bits
        self.shift_register_n_bits = shift_register_n_bits
        self.pool_window = pool_window
        self.n_adder_tree_inputs = n_adder_tree_inputs
        self.n_mux_inputs = n_mux_inputs

        leak_power = self.query_neurosim(self._get_component_name(), self.logger)[
            "Leakage"
        ]
        area = self.query_neurosim(self._get_component_name(), self.logger)["Area"]
        super().__init__(leak_power=leak_power, area=area)

    def build_crossbar(self, overrides: Dict[str, float] = {}):
        cell_config = self.cell_config
        peripheral_args = [
            (k, v) for k, v in self.__dict__.items() if k in PERIPHERAL_PARAMS
        ]
        attrs = {
            "sequential": self.sequential,
            "rows": self.rows,
            "cols": self.cols,
            "cols_muxed": math.ceil(self.cols / self.cols_active_at_once),
            "tech_node": self.tech_node,
            "adc_resolution": self.adc_resolution,
            "read_pulse_width": self.read_pulse_width,
            "cycle_period": self.cycle_period,
            "voltage_dac_bits": self.voltage_dac_bits,
            "temporal_dac_bits": self.temporal_dac_bits,
            "temporal_spiking": self.temporal_spiking,
            "voltage": self.voltage,
            "threshold_voltage": self.threshold_voltage,
            "n_bits": self.n_bits,
            "shift_register_n_bits": self.shift_register_n_bits,
            "pool_window": self.pool_window,
            "n_adder_tree_inputs": self.n_adder_tree_inputs,
            "n_mux_inputs": self.n_mux_inputs,
        }
        attrs.update(overrides)
        key = dict_to_str(attrs)
        if key not in CACHE:
            CACHE[key] = neurointerface.Crossbar(**attrs)
            CACHE[key].run_neurosim(
                cell_config, neurointerface.DEFAULT_CONFIG, self.logger, peripheral_args
            )
        else:
            self.logger.debug(
                "Found cached output for %s. If you're looking for the "
                "log for this, see previous debug messages.",
                key,
            )
        return CACHE[key]

    def get_neurosim_output(
        self, kind: str, adc_resolution: int = None
    ) -> Dict[str, float]:
        """Queries Neurosim for the stats for 'kind' component with 'attributes' attributes"""
        attributes = {
            "cell_config": self.cell_config,
            "tech_node": self.tech_node,
            "cycle_period": self.cycle_period,
            "rows": self.rows,
            "cols": self.cols,
            "cols_active_at_once": self.cols_active_at_once,
            "adc_resolution": (
                adc_resolution if adc_resolution is not None else self.adc_resolution
            ),
            "read_pulse_width": self.read_pulse_width,
            "voltage_dac_bits": self.voltage_dac_bits,
            "temporal_dac_bits": self.temporal_dac_bits,
            "temporal_spiking": self.temporal_spiking,
            "voltage": self.voltage,
            "threshold_voltage": self.threshold_voltage,
            "sequential": self.sequential,
            "average_input_value": self.average_input_value,
            "average_cell_value": self.average_cell_value,
            "n_bits": self.n_bits,
            "shift_register_n_bits": self.shift_register_n_bits,
            "pool_window": self.pool_window,
            "n_adder_tree_inputs": self.n_adder_tree_inputs,
            "n_mux_inputs": self.n_mux_inputs,
        }
        self.logger.debug(
            "Querying Neurosim for %s with attributes: %s", kind, attributes
        )

        # Load defaults
        to_pass = {k: v[1] for k, v in ALL_PARAMS.items()}
        # Get call function ready
        callfunc = self._get_stats_func
        params = {k: ALL_PARAMS[k] for k in self._params}
        docs = {k: v[0] for k, v in params.items()}

        # Get required parameters
        for p in params:
            if "REQUIRED" in params[p][0]:
                assert p in attributes, (
                    f"Failed to generate {kind}. Required parameter not found: "
                    f"{p}. Usage: \n{dict_to_str(docs)}"
                )
            elif p not in attributes:
                attributes[p] = to_pass[p]

            passtype = params[p][2] if len(params[p]) > 2 else int
            try:
                if isinstance(attributes[p], str) and passtype != str:
                    t = "".join(c for c in attributes[p] if (c.isdigit() or c == "."))
                else:
                    t = attributes[p]
                if t != attributes[p]:
                    self.logger.warning(
                        f"WARN: Non-numeric {attributes[p]} for parameter {p}. Using {t} instead."
                    )
                to_pass[p] = passtype(t)
            except ValueError as e:
                raise ValueError(
                    f"Failed to generate {kind}. Parameter {p} must be of type "
                    f'{passtype}. Given: "{attributes[p]}" Usage: \n{dict_to_str(docs)}'
                ) from e

        tn = PERMITTED_TECH_NODES
        assert to_pass["rows"] >= 8, f'Rows must be >=8. Got {to_pass["rows"]}'
        assert to_pass["cols"] >= 8, f'Columns must be >=8. Given: {to_pass["columns"]}'
        assert to_pass["cols_active_at_once"] >= 1, (
            f"Columns active at once must be >=1 and divide evenly into cols. "
            f'Given: {to_pass["cols"]} cols, {to_pass["cols_active_at_once"]} cols active at once'
        )
        assert (
            min(tn) <= to_pass["tech_node"] <= max(tn)
        ), f'Tech node must be between {max(tn)} and {min(tn)}. Given: {to_pass["tech_node"]}'
        assert (
            to_pass["n_bits"] >= 1
        ), f'Adder resolution must be >=1. Given: {to_pass["n_bits"]}'
        assert (
            to_pass["shift_register_n_bits"] >= 1
        ), f'Shift register resolution must be >=1. Given: {to_pass["shift_register_n_bits"]}'
        assert (
            to_pass["pool_window"] >= 1
        ), f'Max pool window size must be >=1. Given: {to_pass["window_size"]}'
        assert (
            to_pass["n_adder_tree_inputs"] > 0
        ), f'Number of adder tree inputs must be >=1. Given: {to_pass["n_adder_tree_inputs"]}'
        assert (
            to_pass["n_mux_inputs"] > 0
        ), f'Number of mux inputs must be >=1. Given: {to_pass["n_mux_inputs"]}'
        assert (
            to_pass["voltage_dac_bits"] > 0
        ), f'Voltage DAC bits must be >=1. Given: {to_pass["voltage_dac_bits"]}'
        assert (
            to_pass["temporal_dac_bits"] > 0
        ), f'Temporal DAC bits must be >=1. Given: {to_pass["temporal_dac_bits"]}'

        if not os.path.exists(to_pass["cell_config"]):
            cell_config = os.path.join(
                SCRIPT_DIR, "cells", to_pass["cell_config"] + ".cell"
            )
            assert os.path.exists(cell_config), (
                f'Cell config {to_pass["cell_config"]}" not found. '
                f'Try a sample config: "{", ".join(SAMPLE_CELLS)}"'
            )
            to_pass["cell_config"] = cell_config

        # Interpolate the tech_node node
        t = to_pass["tech_node"]
        del to_pass["tech_node"]

        for k in attributes:
            if k not in to_pass:
                to_pass[k] = attributes[k]

        hi = min(p for p in PERMITTED_TECH_NODES if p >= t)
        lo = max(p for p in PERMITTED_TECH_NODES if p <= t)
        interp_pt = (t - lo) / (hi - lo) if hi - lo else 0
        hi_crossbar = self.build_crossbar(overrides={"tech_node": hi})
        lo_crossbar = self.build_crossbar(overrides={"tech_node": lo})
        hi_est = callfunc(
            hi_crossbar, to_pass["average_input_value"], to_pass["average_cell_value"]
        )
        lo_est = callfunc(
            lo_crossbar, to_pass["average_input_value"], to_pass["average_cell_value"]
        )
        if hi != lo:
            self.logger.debug(
                "Interpolating between %s and %s. Interpolation " "point: %s",
                lo,
                hi,
                interp_pt,
            )

        rval = {k: lo_est[k] + (hi_est[k] - lo_est[k]) * interp_pt for k in hi_est}
        self.logger.debug("NeuroSim returned: %s", rval)

        assert rval["Area"] >= 0, dedent(
            """
            NeuroSim returned an area less than zero. This may occur if the array or
            memory cell size is too small for proper layout of peripheral
            components. Try increasing the number of rows/columns or increasing the
            cell size.
            """
        )

        # pJ->J, um^2->m^2
        rval = {k: v / 1e12 for k, v in rval.items()}

        return rval

    def query_neurosim(self, kind: str, logger: logging.Logger) -> Dict[str, float]:
        attributes = {
            "cell_config": self.cell_config,
            "tech_node": self.tech_node,
            "cycle_period": self.cycle_period,
            "rows": self.rows,
            "cols": self.cols,
            "cols_active_at_once": self.cols_active_at_once,
            "adc_resolution": self.adc_resolution,
            "read_pulse_width": self.read_pulse_width,
            "voltage_dac_bits": self.voltage_dac_bits,
            "temporal_dac_bits": self.temporal_dac_bits,
            "temporal_spiking": self.temporal_spiking,
            "voltage": self.voltage,
            "threshold_voltage": self.threshold_voltage,
            "sequential": self.sequential,
            "average_input_value": self.average_input_value,
            "average_cell_value": self.average_cell_value,
            "n_bits": self.n_bits,
            "shift_register_n_bits": self.shift_register_n_bits,
            "pool_window": self.pool_window,
            "n_adder_tree_inputs": self.n_adder_tree_inputs,
            "n_mux_inputs": self.n_mux_inputs,
        }

        if kind == "ArrayColDrivers":
            attributes["adc_resolution"] = 0
            return self.get_neurosim_output(kind)

        if kind in ["ArrayADC", "ArrayColDrivers"]:
            logger.info("First running WITH the ADC to get total energy")
            with_adc = self.get_neurosim_output(kind)
            logger.info("Now running WITHOUT the ADC to get column driver energy")
            without_adc = self.get_neurosim_output(kind, adc_resolution=0)
            logger.info("Subtracting column driver energy to get ADC energy")
            return {k: with_adc[k] - without_adc[k] for k in with_adc}
        return self.get_neurosim_output(kind)

    def _get_component_name(self):
        if isinstance(self.component_name, str):
            return self.component_name
        else:
            return self.component_name[0]

    @action
    def read(self) -> tuple[float, float]:
        energy = self.query_neurosim(self._get_component_name(), self.logger)[
            "Read Energy"
        ]
        return energy, 0.0

    @action
    def compute(self) -> tuple[float, float]:
        return self.read()

    @action
    def add(self) -> tuple[float, float]:
        return self.read()

    @action
    def convert(self) -> tuple[float, float]:
        return self.read()

    @action
    def write(self) -> tuple[float, float]:
        energy = self.query_neurosim(self._get_component_name(), self.logger)[
            "Write Energy"
        ]
        return energy, 0.0

    @action
    def update(self) -> tuple[float, float]:
        return self.write()


class NORGate(_NeurosimPlugInComponent):
    """
    A single NOR gate.

    Parameters
    ----------
    tech_node : float
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.

    Attributes
    ----------
    tech_node : float
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    """

    component_name = ["NORGate", "NeuroSimNORGate"]
    _get_stats_func = staticmethod(neurointerface.nor_gate_stats)
    _params = ["tech_node", "cycle_period"]

    def __init__(self, tech_node: float, cycle_period: float):
        super().__init__(
            tech_node=tech_node,
            cycle_period=cycle_period,
        )

    @action
    def read(self) -> tuple[float, float]:
        """
        Returns the energy for one NOR operation in Joules.

        Returns
        -------
        (energy, latency): Tuple in (Joules, seconds).
        """
        return super().read()


class NANDGate(_NeurosimPlugInComponent):
    """
    A single NAND gate.

    Parameters
    ----------
    tech_node : float
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.

    Attributes
    ----------
    tech_node : float
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    """

    component_name = ["NANDGate", "NeuroSimNANDGate"]
    _get_stats_func = staticmethod(neurointerface.nand_gate_stats)
    _params = ["tech_node", "cycle_period"]

    def __init__(self, tech_node: float, cycle_period: float):
        super().__init__(
            tech_node=tech_node,
            cycle_period=cycle_period,
        )

    @action
    def read(self) -> tuple[float, float]:
        """
        Returns the energy and latency for one NAND operation.
        """
        return super().read()


class NOTGate(_NeurosimPlugInComponent):
    """
    A single NOT gate.

    Parameters
    ----------
    tech_node : float
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.

    Attributes
    ----------
    tech_node : float
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    """

    component_name = ["NOTGate", "NeuroSimNOTGate"]
    _get_stats_func = staticmethod(neurointerface.not_gate_stats)
    _params = ["tech_node", "cycle_period"]

    def __init__(self, tech_node: float, cycle_period: float):
        super().__init__(
            tech_node=tech_node,
            cycle_period=cycle_period,
        )

    @action
    def read(self) -> tuple[float, float]:
        """
        Returns the energy and latency for one NOT operation.
        """
        return super().read()


class FlipFlop(_NeurosimPlugInComponent):
    """
    A single flip-flop. Stores one bit of data.

    Parameters
    ----------
    tech_node : float
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    n_bits : int
        The number of bits, and therefore the number of flip-flops.

    Attributes
    ----------
    tech_node : float
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    n_bits : int
        The number of bits, and therefore the number of flip-flops.
    """

    component_name = ["FlipFlop", "NeuroSimFlipFlop"]
    _get_stats_func = staticmethod(neurointerface.flip_flop_stats)
    _params = ["tech_node", "cycle_period", "n_bits"]

    def __init__(self, tech_node: float, cycle_period: float, n_bits: int):
        super().__init__(
            tech_node=tech_node,
            cycle_period=cycle_period,
            n_bits=n_bits,
        )

    def read(self) -> tuple[float, float]:
        """
        Returns the energy and latency for one flip-flop read operation.
        """
        return super().read()

    def write(self) -> tuple[float, float]:
        """
        Returns the energy and latency for one flip-flop write operation.
        """
        return super().write()


class Mux(_NeurosimPlugInComponent):
    """
    A single mux. Selects one of multiple inputs.

    Parameters
    ----------
    tech_node : float
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    n_bits : int
        The number of bits for each of the mux's inputs.
    n_mux_inputs : int
        The number of inputs to the mux.

    Attributes
    ----------
    tech_node : float
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    n_bits : int
        The number of bits for each of the mux's inputs.
    n_mux_inputs : int
        The number of inputs to the mux.
    """

    component_name = ["Mux", "NeuroSimMux"]
    _get_stats_func = staticmethod(neurointerface.mux_stats)
    _params = ["tech_node", "cycle_period", "n_bits", "n_mux_inputs"]

    def __init__(
        self,
        tech_node: float,
        cycle_period: float,
        n_bits: int,
        n_mux_inputs: int,
    ):
        super().__init__(
            tech_node=tech_node,
            cycle_period=cycle_period,
            n_bits=n_bits,
            n_mux_inputs=n_mux_inputs,
        )

    def read(self) -> tuple[float, float]:
        """
        Returns the energy and latency for one muxing operation.
        """
        return super().read()


class Adder(_NeurosimPlugInComponent):
    """
    A single adder. Sums two values into one.

    Parameters
    ----------
    tech_node : float
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    n_bits : int
        The number of bits of the adder's inputs.

    Attributes
    ----------
    tech_node : float
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    n_bits : int
        The number of bits of the adder's inputs.
    """

    component_name = ["Adder", "NeuroSimAdder", "IntAdder"]
    _get_stats_func = staticmethod(neurointerface.adder_stats)
    _params = ["tech_node", "cycle_period", "n_bits"]

    def __init__(self, tech_node: float, cycle_period: float, n_bits: int):
        super().__init__(
            tech_node=tech_node,
            cycle_period=cycle_period,
            n_bits=n_bits,
        )

    def add(self) -> tuple[float, float]:
        """
        Returns the energy for one addition operation in Joules.

        Returns
        -------
        (energy, latency): Tuple in (Joules, seconds).
        """
        return super().add()

    def read(self) -> tuple[float, float]:
        """
        Returns the energy for one addition operation in Joules.

        Returns
        -------
        (energy, latency): Tuple in (Joules, seconds).
        """
        return super().read()


class AdderTree(_NeurosimPlugInComponent):
    """
    An adder tree. Sums multiple (>2) values into one.

    Parameters
    ----------
    tech_node : float
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    n_bits : int
        The number of bits of the adder tree's inputs.
    n_adder_tree_inputs : int
        The number of values added by the adder tree.

    Attributes
    ----------
    tech_node : float
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    n_bits : int
        The number of bits of the adder tree's inputs.
    n_adder_tree_inputs : int
        The number of values added by the adder tree.
    """

    component_name = ["AdderTree", "NeuroSimAdderTree", "IntAdderTree"]
    _get_stats_func = staticmethod(neurointerface.adder_tree_stats)
    _params = ["tech_node", "cycle_period", "n_bits", "n_adder_tree_inputs"]

    def __init__(
        self,
        tech_node: float,
        cycle_period: float,
        n_bits: int,
        n_adder_tree_inputs: int,
    ):
        super().__init__(
            tech_node=tech_node,
            cycle_period=cycle_period,
            n_bits=n_bits,
            n_adder_tree_inputs=n_adder_tree_inputs,
        )

    @action
    def add(self) -> tuple[float, float]:
        """
        Returns the energy for one addition operation in Joules.

        Returns
        -------
        (energy, latency): Tuple in (Joules, seconds).
        """
        return super().add()

    @action
    def read(self) -> tuple[float, float]:
        """
        Returns the energy for one addition operation in Joules.

        Returns
        -------
        (energy, latency): Tuple in (Joules, seconds).
        """
        return super().read()


class MaxPool(_NeurosimPlugInComponent):
    """
    A max pool unit.

    Parameters
    ----------
    tech_node : float
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    n_bits : int
        The number of bits of the max pool unit's inputs.
    pool_window : int
        The window size of max pooling.

    Attributes
    ----------
    tech_node : float
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    n_bits : int
        The number of bits of the max pool unit's inputs.
    pool_window : int
        The window size of max pooling.
    """

    component_name = ["MaxPool", "NeuroSimMaxPool"]
    _get_stats_func = staticmethod(neurointerface.max_pool_stats)
    _params = ["tech_node", "cycle_period", "n_bits", "pool_window"]

    def __init__(
        self, tech_node: float, cycle_period: float, n_bits: int, pool_window: int
    ):
        super().__init__(
            tech_node=tech_node,
            cycle_period=cycle_period,
            n_bits=n_bits,
            pool_window=pool_window,
        )


class ShiftAdd(_NeurosimPlugInComponent):
    """
    A shift-and-add unit. This unit will sum and accumulate values in a register, while
    also shifting the register contents to accept various power-of-two scaling factors
    for the summed values.

    Parameters
    ----------
    tech_node : float
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    n_bits : int
        The number of bits of the shift-and-add unit's inputs.
    shift_register_n_bits : int
        The number of bits of the shift register.

    Attributes
    ----------
    tech_node : float
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    n_bits : int
        The number of bits of the shift-and-add unit's inputs.
    shift_register_n_bits : int
        The number of bits of the shift register.
    """

    component_name = ["ShiftAdd", "NeuroSimShiftAdd"]
    _get_stats_func = staticmethod(neurointerface.shift_add_stats)
    _params = ["tech_node", "cycle_period", "n_bits", "shift_register_n_bits"]

    def __init__(
        self,
        tech_node: float,
        cycle_period: float,
        n_bits: int,
        shift_register_n_bits: int,
    ):
        super().__init__(
            tech_node=tech_node,
            cycle_period=cycle_period,
            n_bits=n_bits,
            shift_register_n_bits=shift_register_n_bits,
        )

    @action
    def read(self) -> tuple[float, float]:
        """
        Returns the energy and latency to read the shift-and-add unit's output.
        """
        return super().read()

    @action
    def write(self) -> tuple[float, float]:
        """
        Returns the energy and latency to shift-and-add.
        """
        return super().shift_add()

    @action
    def shift_add(self) -> tuple[float, float]:
        """
        Returns the energy and latency to shift-and-add.
        """
        return super().shift_add()


class _NeurosimPIMComponent(_NeurosimPlugInComponent):
    """
    Base class for Neurosim PIM (Processing In Memory) and/or CiM (Computing In Memory)
    components.

    Parameters
    ----------
    cell_config : str
        The path to the  cell config file to use.
    tech_node : str
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    rows : int
        The number of rows in the CiM array.
    cols : int
        The number of columns in the CiM array.
    cols_active_at_once : int
        The number of columns active at once. If this is less than the number of
        columns, the array is activated in multiple steps.
    adc_resolution : int, optional
        The resolution of the ADC. Resolution of 0 means no ADC, and the CiM array
        outputs leave as analog signals.
    read_pulse_width : float, optional
        The width of the read pulse in seconds. This is the duration for which the row
        is asserted when reading from the CiM array.
    voltage_dac_bits : int, optional
        The number of bits of a voltage DAC, which outputs analog signals by setting
        voltage to a given value.
    temporal_dac_bits : int, optional
        The number of bits of a temporal DAC, which outputs analog signals by setting
        the pulse width of a signal. Read pulse width will be multiplied by
        (2^temporal_dac_bits - 1).
    temporal_spiking : bool, optional
        Whether to use a spiking (#pulses) or a PWM (pulse length) approach for temporal
        DAC.
    voltage : float, optional
        The supply voltage in volts.
    threshold_voltage : float, optional
        The threshold voltage of all transistors, in volts.
    sequential : bool, optional
        Whether to use a sequential approach for the CiM array, meaing that rows are
        activated one at a time.
    average_input_value : float, optional
        The average input value to a row. Must be between 0 and 1.
    average_cell_value : float, optional
        The average value of a cell. Must be between 0 and 1.
    """

    component_name = "_override_this_name_"
    _params = [
        "tech_node",
        "cycle_period",
        "rows",
        "cols",
        "cols_active_at_once",
        "cell_config",
        "average_input_value",
        "average_cell_value",
        "read_pulse_width",
        "adc_resolution",
        "voltage_dac_bits",
        "temporal_dac_bits",
        "temporal_spiking",
        "voltage",
        "threshold_voltage",
        "sequential",
    ]

    def __init__(
        self,
        tech_node: float,
        cycle_period: float,
        rows: int,
        cols: int,
        cols_active_at_once: int,
        cell_config: str,
        average_input_value: float,
        average_cell_value: float,
        read_pulse_width: float,
        adc_resolution: int = 0,
        voltage_dac_bits: int = 1,
        temporal_dac_bits: int = 1,
        temporal_spiking: bool = True,
        voltage: float = 0,
        threshold_voltage: float = 0,
        sequential: bool = False,
    ):
        super().__init__(
            tech_node=tech_node,
            cycle_period=cycle_period,
            rows=rows,
            cols=cols,
            cols_active_at_once=cols_active_at_once,
            cell_config=cell_config,
            average_input_value=average_input_value,
            average_cell_value=average_cell_value,
            read_pulse_width=read_pulse_width,
            adc_resolution=adc_resolution,
            voltage_dac_bits=voltage_dac_bits,
            temporal_dac_bits=temporal_dac_bits,
            temporal_spiking=temporal_spiking,
            voltage=voltage,
            threshold_voltage=threshold_voltage,
            sequential=sequential,
        )


class RowDrivers(_NeurosimPIMComponent):
    """
    Row drivers for a CiM array, which select array rows and raise them to a read
    voltage.

    Parameters
    ----------
    cell_config : str
        The path to the  cell config file to use.
    tech_node : str
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    rows : int
        The number of rows in the CiM array.
    cols : int
        The number of columns in the CiM array.
    cols_active_at_once : int
        The number of columns active at once. If this is less than the number of
        columns, the array is activated in multiple steps.
    adc_resolution : int, optional
        The resolution of the ADC. Resolution of 0 means no ADC, and the CiM array
        outputs leave as analog signals.
    read_pulse_width : float, optional
        The width of the read pulse in seconds. This is the duration for which the row
        is asserted when reading from the CiM array.
    voltage_dac_bits : int, optional
        The number of bits of a voltage DAC, which outputs analog signals by setting
        voltage to a given value.
    temporal_dac_bits : int, optional
        The number of bits of a temporal DAC, which outputs analog signals by setting
        the pulse width of a signal. Read pulse width will be multiplied by
        (2^temporal_dac_bits - 1).
    temporal_spiking : bool, optional
        Whether to use a spiking (#pulses) or a PWM (pulse length) approach for temporal
        DAC.
    voltage : float, optional
        The supply voltage in volts.
    threshold_voltage : float, optional
        The threshold voltage of all transistors, in volts.
    sequential : bool, optional
        Whether to use a sequential approach for the CiM array, meaing that rows are
        activated one at a time.
    average_input_value : float, optional
        The average input value to a row. Must be between 0 and 1.
    average_cell_value : float, optional
        The average value of a cell. Must be between 0 and 1.
    """

    component_name = ["ArrayRowDrivers", "NeuroSimArrayRowDrivers"]
    _get_stats_func = staticmethod(neurointerface.row_stats)


class ColDrivers(_NeurosimPIMComponent):
    """
    Column drivers for a CiM array, which precharge array coolumns and select them for
    reading.

    Parameters
    ----------
    cell_config : str
        The path to the  cell config file to use.
    tech_node : str
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    rows : int
        The number of rows in the CiM array.
    cols : int
        The number of columns in the CiM array.
    cols_active_at_once : int
        The number of columns active at once. If this is less than the number of
        columns, the array is activated in multiple steps.
    adc_resolution : int, optional
        The resolution of the ADC. Resolution of 0 means no ADC, and the CiM array
        outputs leave as analog signals.
    read_pulse_width : float, optional
        The width of the read pulse in seconds. This is the duration for which the row
        is asserted when reading from the CiM array.
    voltage_dac_bits : int, optional
        The number of bits of a voltage DAC, which outputs analog signals by setting
        voltage to a given value.
    temporal_dac_bits : int, optional
        The number of bits of a temporal DAC, which outputs analog signals by setting
        the pulse width of a signal. Read pulse width will be multiplied by
        (2^temporal_dac_bits - 1).
    temporal_spiking : bool, optional
        Whether to use a spiking (#pulses) or a PWM (pulse length) approach for temporal
        DAC.
    voltage : float, optional
        The supply voltage in volts.
    threshold_voltage : float, optional
        The threshold voltage of all transistors, in volts.
    sequential : bool, optional
        Whether to use a sequential approach for the CiM array, meaing that rows are
        activated one at a time.
    average_input_value : float, optional
        The average input value to a row. Must be between 0 and 1.
    average_cell_value : float, optional
        The average value of a cell. Must be between 0 and 1.
    """

    component_name = [
        "ArrayColDrivers",
        "NeuroSimArrayColDrivers",
        "ArrayColumnDrivers",
        "NeuroSimArrayColumnDrivers",
    ]
    _get_stats_func = staticmethod(neurointerface.col_stats)


class ADC(_NeurosimPIMComponent):
    """
    Analog-digital-converters (ADCs) for a CiM array, which read analog column outputs
    and convert them to digital values.

    Parameters
    ----------
    cell_config : str
        The path to the  cell config file to use.
    tech_node : str
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    rows : int
        The number of rows in the CiM array.
    cols : int
        The number of columns in the CiM array.
    cols_active_at_once : int
        The number of columns active at once. If this is less than the number of
        columns, the array is activated in multiple steps.
    adc_resolution : int, optional
        The resolution of the ADC. Resolution of 0 means no ADC, and the CiM array
        outputs leave as analog signals.
    read_pulse_width : float, optional
        The width of the read pulse in seconds. This is the duration for which the row
        is asserted when reading from the CiM array.
    voltage_dac_bits : int, optional
        The number of bits of a voltage DAC, which outputs analog signals by setting
        voltage to a given value.
    temporal_dac_bits : int, optional
        The number of bits of a temporal DAC, which outputs analog signals by setting
        the pulse width of a signal. Read pulse width will be multiplied by
        (2^temporal_dac_bits - 1).
    temporal_spiking : bool, optional
        Whether to use a spiking (#pulses) or a PWM (pulse length) approach for temporal
        DAC.
    voltage : float, optional
        The supply voltage in volts.
    threshold_voltage : float, optional
        The threshold voltage of all transistors, in volts.
    sequential : bool, optional
        Whether to use a sequential approach for the CiM array, meaing that rows are
        activated one at a time.
    average_input_value : float, optional
        The average input value to a row. Must be between 0 and 1.
    average_cell_value : float, optional
        The average value of a cell. Must be between 0 and 1.
    """

    component_name = ["ArrayADC", "NeuroSimArrayADC"]
    _get_stats_func = staticmethod(neurointerface.col_stats)


class MemoryCell(_NeurosimPIMComponent):
    """
    A memory cell for a CiM array, which stores data and performs analog computation.

    Parameters
    ----------
    cell_config : str
        The path to the  cell config file to use.
    tech_node : str
        The technology node in meters.
    cycle_period : float
        The time period of the system clock in seconds.
    rows : int
        The number of rows in the CiM array.
    cols : int
        The number of columns in the CiM array.
    cols_active_at_once : int
        The number of columns active at once. If this is less than the number of
        columns, the array is activated in multiple steps.
    adc_resolution : int, optional
        The resolution of the ADC. Resolution of 0 means no ADC, and the CiM array
        outputs leave as analog signals.
    read_pulse_width : float, optional
        The width of the read pulse in seconds. This is the duration for which the row
        is asserted when reading from the CiM array.
    voltage_dac_bits : int, optional
        The number of bits of a voltage DAC, which outputs analog signals by setting
        voltage to a given value.
    temporal_dac_bits : int, optional
        The number of bits of a temporal DAC, which outputs analog signals by setting
        the pulse width of a signal. Read pulse width will be multiplied by
        (2^temporal_dac_bits - 1).
    temporal_spiking : bool, optional
        Whether to use a spiking (#pulses) or a PWM (pulse length) approach for temporal
        DAC.
    voltage : float, optional
        The supply voltage in volts.
    threshold_voltage : float, optional
        The threshold voltage of all transistors, in volts.
    sequential : bool, optional
        Whether to use a sequential approach for the CiM array, meaing that rows are
        activated one at a time.
    average_input_value : float, optional
        The average input value to a row. Must be between 0 and 1.
    average_cell_value : float, optional
        The average value of a cell. Must be between 0 and 1.
    """

    component_name = ["MemoryCell", "NeuroSimMemoryCell"]
    _get_stats_func = staticmethod(neurointerface.cell_stats)


# ==================================================================================================
# Input Parsing
# ==================================================================================================


def build_crossbar(
    attrs: dict, logger: logging.Logger, overrides: dict = {}
) -> neurointerface.Crossbar:
    """Builds a crossbar from the given attributes"""
    cell_config = attrs["cell_config"]
    peripheral_args = [(k, v) for k, v in attrs.items() if k in PERIPHERAL_PARAMS]
    attrs = {
        "sequential": attrs["sequential"],
        "rows": attrs["rows"],
        "cols": attrs["cols"],
        "cols_muxed": math.ceil(attrs["cols"] / attrs["cols_active_at_once"]),
        "tech_node": attrs["tech_node"],
        "adc_resolution": attrs[f"adc_resolution"],
        "read_pulse_width": attrs["read_pulse_width"],
        "cycle_period": attrs["cycle_period"],
        "voltage_dac_bits": attrs["voltage_dac_bits"],
        "temporal_dac_bits": attrs["temporal_dac_bits"],
        "temporal_spiking": attrs["temporal_spiking"],
        "voltage": attrs["voltage"],
        "threshold_voltage": attrs["threshold_voltage"],
        "n_bits": attrs["n_bits"],
        "shift_register_n_bits": attrs["shift_register_n_bits"],
        "pool_window": attrs["pool_window"],
        "n_adder_tree_inputs": attrs["n_adder_tree_inputs"],
        "n_mux_inputs": attrs["n_mux_inputs"],
    }
    attrs.update(overrides)
    key = dict_to_str(attrs)
    if key not in CACHE:
        CACHE[key] = neurointerface.Crossbar(**attrs)
        CACHE[key].run_neurosim(
            cell_config, neurointerface.DEFAULT_CONFIG, logger, peripheral_args
        )
    else:
        logger.debug(
            "Found cached output for %s. If you're looking for the "
            "log for this, see previous debug messages.",
            key,
        )
    return CACHE[key]


def dict_to_str(attributes: Dict) -> str:
    """Converts a dictionary into a multi-line string representation"""
    s = "\n"
    for k, v in attributes.items():
        s += f"\t{k}: {v}\n"
    return s
