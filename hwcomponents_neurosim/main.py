from abc import ABC
import math
import sys
import os
from typing import Dict
from textwrap import dedent
from hwcomponents import EnergyAreaEstimator, actionDynamicEnergy
import hwcomponents_neurosim.neurointerface as neurointerface

# ==================================================================================================
# Constants
# ==================================================================================================
SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
DEBUG = False
ACCURACY = 70
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
    "global_cycle_seconds": (
        f"REQUIRED: Duration of one cycle in seconds",
        1e-9,
        float,
    ),
    "tech_node": (
        f"REQUIRED: tech_node node. Must be between {max(PERMITTED_TECH_NODES)}  "
        f"and {min(PERMITTED_TECH_NODES)}.",
        32,
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

logger = None


class NeurosimPlugInComponent(EnergyAreaEstimator):
    component_name = "_override_this_name_"
    percent_accuracy_0_to_100 = 70
    _get_stats_func = None  # To be overridden by child classes
    _params = []  # To be overridden by child classes

    def __init__(
        self,
        cell_config: str,
        tech_node: str,
        global_cycle_seconds: float = 100e-9,
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
        self.global_cycle_seconds = global_cycle_seconds
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

        leakage_power = self.query_neurosim(self._get_component_name())["Leakage"]
        area = self.query_neurosim(self._get_component_name())["Area"]
        super().__init__(leakage_power=leakage_power, area=area)

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
            "global_cycle_seconds": self.global_cycle_seconds,
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
                cell_config, neurointerface.DEFAULT_CONFIG, peripheral_args
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
            "global_cycle_seconds": self.global_cycle_seconds,
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
        params = self._params
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
        return rval

    def query_neurosim(self, kind: str) -> Dict[str, float]:
        attributes = {
            "cell_config": self.cell_config,
            "tech_node": self.tech_node,
            "global_cycle_seconds": self.global_cycle_seconds,
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

        if kind == "array_col_drivers":
            attributes["adc_resolution"] = 0
            return self.get_neurosim_output(kind)

        if kind in ["array_adc", "array_col_drivers"]:
            logger.info("First running WITH the ADC to get total energy")
            with_adc = self.get_neurosim_output(kind)
            logger.info("Now running WITHOUT the ADC to get column driver energy")
            without_adc = self.get_neurosim_output(kind, adc_resolution=0)
            logger.info("Subtracting column driver energy to get ADC energy")
            return {k: with_adc[k] - without_adc[k] for k in with_adc}
        return self.get_neurosim_output(kind)

    def _get_component_name(self):
        if isinstance(self.name, str):
            return self.name
        else:
            return self.name[0]

    @actionDynamicEnergy
    def read(self):
        return self.query_neurosim(self._get_component_name())["Read Energy"]

    @actionDynamicEnergy
    def compute(self):
        return self.read()

    @actionDynamicEnergy
    def add(self):
        return self.read()

    @actionDynamicEnergy
    def shift_add(self):
        return self.read()

    @actionDynamicEnergy
    def convert(self):
        return self.read()

    @actionDynamicEnergy
    def write(self):
        return self.query_neurosim(self._get_component_name())["Write Energy"]

    @actionDynamicEnergy
    def update(self):
        return self.write()


class NORGate(NeurosimPlugInComponent):
    component_name = "nor_gate"
    _get_stats_func = neurointerface.nor_gate_stats
    _params = ["tech_node", "global_cycle_seconds"]

    def __init__(self, tech_node: str, global_cycle_seconds: float):
        super().__init__(
            tech_node=tech_node,
            global_cycle_seconds=global_cycle_seconds,
        )


class NANDGate(NeurosimPlugInComponent):
    component_name = "nand_gate"
    _get_stats_func = neurointerface.nand_gate_stats
    _params = ["tech_node", "global_cycle_seconds"]

    def __init__(self, tech_node: str, global_cycle_seconds: float):
        super().__init__(
            tech_node=tech_node,
            global_cycle_seconds=global_cycle_seconds,
        )


class NOTGate(NeurosimPlugInComponent):
    component_name = "not_gate"
    _get_stats_func = neurointerface.not_gate_stats
    _params = ["tech_node", "global_cycle_seconds"]

    def __init__(self, tech_node: str, global_cycle_seconds: float):
        super().__init__(
            tech_node=tech_node,
            global_cycle_seconds=global_cycle_seconds,
        )


class FlipFlop(NeurosimPlugInComponent):
    component_name = "flip_flop"
    _get_stats_func = neurointerface.flip_flop_stats
    _params = ["tech_node", "global_cycle_seconds", "n_bits"]

    def __init__(self, tech_node: str, global_cycle_seconds: float, n_bits: int):
        super().__init__(
            tech_node=tech_node,
            global_cycle_seconds=global_cycle_seconds,
            n_bits=n_bits,
        )


class Mux(NeurosimPlugInComponent):
    component_name = "mux"
    _get_stats_func = neurointerface.mux_stats
    _params = ["tech_node", "global_cycle_seconds", "n_bits", "n_mux_inputs"]

    def __init__(
        self,
        tech_node: str,
        global_cycle_seconds: float,
        n_bits: int,
        n_mux_inputs: int,
    ):
        super().__init__(
            tech_node=tech_node,
            global_cycle_seconds=global_cycle_seconds,
            n_bits=n_bits,
            n_mux_inputs=n_mux_inputs,
        )


class Adder(NeurosimPlugInComponent):
    component_name = "adder"
    _get_stats_func = neurointerface.adder_stats
    _params = ["tech_node", "global_cycle_seconds", "n_bits"]

    def __init__(self, tech_node: str, global_cycle_seconds: float, n_bits: int):
        super().__init__(
            tech_node=tech_node,
            global_cycle_seconds=global_cycle_seconds,
            n_bits=n_bits,
        )


class AdderTree(NeurosimPlugInComponent):
    component_name = "adder_tree"
    _get_stats_func = neurointerface.adder_tree_stats
    _params = ["tech_node", "global_cycle_seconds", "n_bits", "n_adder_tree_inputs"]

    def __init__(
        self,
        tech_node: str,
        global_cycle_seconds: float,
        n_bits: int,
        n_adder_tree_inputs: int,
    ):
        super().__init__(
            tech_node=tech_node,
            global_cycle_seconds=global_cycle_seconds,
            n_bits=n_bits,
            n_adder_tree_inputs=n_adder_tree_inputs,
        )


class MaxPool(NeurosimPlugInComponent):
    component_name = "max_pool"
    _get_stats_func = neurointerface.max_pool_stats
    _params = ["tech_node", "global_cycle_seconds", "n_bits", "pool_window"]

    def __init__(
        self, tech_node: str, global_cycle_seconds: float, n_bits: int, pool_window: int
    ):
        super().__init__(
            tech_node=tech_node,
            global_cycle_seconds=global_cycle_seconds,
            n_bits=n_bits,
            pool_window=pool_window,
        )


class ShiftAdd(NeurosimPlugInComponent):
    component_name = "shift_add"
    _get_stats_func = neurointerface.shift_add_stats
    _params = ["tech_node", "global_cycle_seconds", "n_bits", "shift_register_n_bits"]

    def __init__(
        self,
        tech_node: str,
        global_cycle_seconds: float,
        n_bits: int,
        shift_register_n_bits: int,
    ):
        super().__init__(
            tech_node=tech_node,
            global_cycle_seconds=global_cycle_seconds,
            n_bits=n_bits,
            shift_register_n_bits=shift_register_n_bits,
        )


class NeurosimPIMComponent(NeurosimPlugInComponent):
    component_name = "_override_this_name_"
    _params = [
        "tech_node",
        "global_cycle_seconds",
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
        tech_node: str,
        global_cycle_seconds: float,
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
            global_cycle_seconds=global_cycle_seconds,
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


class RowDrivers(NeurosimPIMComponent):
    component_name = "array_row_drivers"
    _get_stats_func = neurointerface.row_stats


class ColDrivers(NeurosimPIMComponent):
    component_name = "array_col_drivers"
    _get_stats_func = neurointerface.col_stats


class ADC(NeurosimPIMComponent):
    component_name = "array_adc"
    _get_stats_func = neurointerface.col_stats


class MemoryCell(NeurosimPIMComponent):
    component_name = "memory_cell"
    _get_stats_func = neurointerface.cell_stats


# ==================================================================================================
# Input Parsing
# ==================================================================================================


def build_crossbar(attrs: dict, overrides: dict = {}) -> neurointerface.Crossbar:
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
        "global_cycle_seconds": attrs["global_cycle_seconds"],
        "voltage_dac_bits": attrs["voltage_dac_bits"],
        "temporal_dac_bits": attrs["temporal_dac_bits"],
        "temporal_spiking": attrs["temporal_spiking"],
        "voltage": attrs["voltage"],
        "threshold_voltage": attrs["threshold_voltage"],
    }
    attrs.update(overrides)
    key = dict_to_str(attrs)
    if key not in CACHE:
        CACHE[key] = neurointerface.Crossbar(**attrs)
        CACHE[key].run_neurosim(
            cell_config, neurointerface.DEFAULT_CONFIG, peripheral_args
        )
    else:
        logger.debug(
            "Found cached output for %s. If you're looking for the "
            "log for this, see previous debug messages.",
            key,
        )
    return CACHE[key]


def get_neurosim_output(kind: str, attributes: dict) -> Dict[str, float]:
    """Queries Neurosim for the stats for 'kind' component with 'attributes' attributes"""
    assert kind in SUPPORTED_CLASSES, f"Unsupported primitive: {kind}"
    logger.debug("Querying Neurosim for %s with attributes: %s", kind, attributes)

    # Load defaults
    to_pass = {k: v[1] for k, v in ALL_PARAMS.items()}
    # Get call function ready
    callfunc = SUPPORTED_CLASSES[kind][0]
    params = SUPPORTED_CLASSES[kind][1]
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
                logger.warning(
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
            f'Try a sample config: "{", ".join(SAMPLE_CELLS)}'
        )
        to_pass["cell_config"] = cell_config

    # Interpolate the tech_node node. If p is in PERMITTED_TECH_NODES, then all this comes out
    # to just p. If p is not in PERMITTED_TECH_NODES, then we interpolate between the two closest.
    t = to_pass["tech_node"]
    del to_pass["tech_node"]

    for k in attributes:
        if k not in to_pass:
            to_pass[k] = attributes[k]

    hi = min(p for p in PERMITTED_TECH_NODES if p >= t)
    lo = max(p for p in PERMITTED_TECH_NODES if p <= t)
    interp_pt = (t - lo) / (hi - lo) if hi - lo else 0
    hi_crossbar = build_crossbar(overrides={"tech_node": hi})
    lo_crossbar = build_crossbar(overrides={"tech_node": lo})
    hi_est = callfunc(
        hi_crossbar, to_pass["average_input_value"], to_pass["average_cell_value"]
    )
    lo_est = callfunc(
        lo_crossbar, to_pass["average_input_value"], to_pass["average_cell_value"]
    )
    if hi != lo:
        logger.debug(
            "Interpolating between %s and %s. Interpolation " "point: %s",
            lo,
            hi,
            interp_pt,
        )

    rval = {k: lo_est[k] + (hi_est[k] - lo_est[k]) * interp_pt for k in hi_est}
    logger.debug("NeuroSim returned: %s", rval)

    assert rval["Area"] >= 0, dedent(
        """
        NeuroSim returned an area less than zero. This may occur if the array or
        memory cell size is too small for proper layout of peripheral
        components. Try increasing the number of rows/columns or increasing the
        cell size.
        """
    )

    return rval


def query_neurosim(kind: str, attributes: dict) -> Dict[str, float]:
    for n in ["array_adc", "array_col_drivers"]:
        assert (
            n in SUPPORTED_CLASSES
        ), "Please update this method body to support the new NeuroSim names."

    if kind == "array_col_drivers":
        attributes["adc_resolution"] = 0
        return get_neurosim_output(kind, attributes)

    if kind in ["array_adc", "array_col_drivers"]:
        logger.info("First running WITH the ADC to get total energy")
        with_adc = get_neurosim_output(kind, attributes)
        attributes["adc_resolution"] = 0
        logger.info("Now running WITHOUT the ADC to get column driver energy")
        without_adc = get_neurosim_output(kind, attributes)
        logger.info("Subtracting column driver energy to get ADC energy")
        return {k: with_adc[k] - without_adc[k] for k in with_adc}
    return get_neurosim_output(kind, attributes)


def dict_to_str(attributes: Dict) -> str:
    """Converts a dictionary into a multi-line string representation"""
    s = "\n"
    for k, v in attributes.items():
        s += f"\t{k}: {v}\n"
    return s
