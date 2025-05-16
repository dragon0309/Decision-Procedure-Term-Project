from pyverilog.vparser.parser import parse
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.vparser.ast import *
import networkx as nx
import json
import os
import subprocess
import warnings
import shutil
from typing import Dict, List, Set, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import graphviz
from .gadget_transformer import GadgetTransformer
from .cnf_encoder import CNFEncoder
from .fault_model import FaultModel, FaultType, GateType

@dataclass
class GateInfo:
    """Information about a gate in the circuit."""
    id: str
    type: str
    instance_name: str
    inputs: List[str]
    output: str
    category: str
    origin_module: str  # Track which module this gate came from
    bit_width: int = 1
    bit_range: Optional[tuple] = None
    clock: Optional[str] = None  # Track clock signal for sequential gates
    initial_state: Optional[Any] = None  # Track initial state for memory gates

GATE_PORT_MAP = {
    'and': {'inputs': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], 'output': ['Y']},
    'or': {'inputs': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], 'output': ['Y']},
    'xor': {'inputs': ['A', 'B'], 'output': ['Y']},
    'not': {'inputs': ['A'], 'output': ['Y']},
    'nand': {'inputs': ['A', 'B', 'C', 'D'], 'output': ['Y']},
    'nor': {'inputs': ['A', 'B', 'C', 'D'], 'output': ['Y']},
    'buf': {'inputs': ['A'], 'output': ['Y']},
    'mux': {'inputs': ['A', 'B', 'S'], 'output': ['Y']},
    'dff': {'inputs': ['D', 'CLK'], 'output': ['Q', 'QN'], 'clock': ['CLK', 'CK', 'C']},
    'dffe': {'inputs': ['D', 'CLK', 'EN'], 'output': ['Q', 'QN'], 'clock': ['CLK', 'CK', 'C']},
    'dffs': {'inputs': ['D', 'CLK', 'SET'], 'output': ['Q', 'QN'], 'clock': ['CLK', 'CK', 'C']},
    'dffr': {'inputs': ['D', 'CLK', 'RESET'], 'output': ['Q', 'QN'], 'clock': ['CLK', 'CK', 'C']},
    'dffrse': {'inputs': ['D', 'CLK', 'RESET', 'SET', 'EN'], 'output': ['Q', 'QN'], 'clock': ['CLK', 'CK', 'C']},
    'latch': {'inputs': ['D', 'EN'], 'output': ['Q'], 'clock': ['EN', 'G']},
}

YOSYS_GATE_MAP = {
    '$and': 'and',
    '$or': 'or',
    '$xor': 'xor',
    '$not': 'not',
    '$buf': 'buf',
    '$nand': 'nand',
    '$nor': 'nor',
    '$mux': 'mux',
    '$dff': 'dff',
    '$dffe': 'dffe',
    '$dffs': 'dffs',
    '$dffr': 'dffr',
    '$dffrse': 'dffrse',
    '$dlatch': 'latch',
    '$adff': 'dff',
}

CLOCK_SIGNAL_NAMES = [
    'clk', 'clock', 'ck', 'c',
    'CLK', 'CLOCK', 'CK', 'C',
    'clk_i', 'clock_i', 'clkin', 'clk_in',
    'CLK_I', 'CLOCK_I', 'CLKIN', 'CLK_IN',
    'pclk', 'sysclk', 'aclk', 'bclk',
    'PCLK', 'SYSCLK', 'ACLK', 'BCLK'
]

class VerilogParser:
    """Parses Verilog files into an AST using Pyverilog."""
    def __init__(self):
        self.ast = None
        self.modules = {}
        self.temp_files = []
        self.module_inputs = {}
        self.module_outputs = {}
        
    def parse_file(self, filename: str, use_synthesis: bool = True) -> None:
        """
        Parse a Verilog file into an AST.
        
        Args:
            filename: Path to the Verilog file to parse
            use_synthesis: Whether to synthesize behavioral Verilog to gate-level netlist first
        """
        if use_synthesis:
            try:
                output_file = self._synthesize_verilog(filename)
                self._parse_verilog_file(output_file)
                self.temp_files.append(output_file)
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                warnings.warn(f"Synthesis failed, falling back to direct parsing. Error: {e}\n"
                              f"Ensure YoWASP Yosys is installed via 'pip install yowasp-yosys' for better behavioral Verilog support.")
                self._parse_verilog_file(filename)
        else:
            self._parse_verilog_file(filename)
            
    def _synthesize_verilog(self, filename: str) -> str:
        """
        Synthesize behavioral Verilog to gate-level netlist using Yosys.
        
        Args:
            filename: Path to input Verilog file
            
        Returns:
            Path to synthesized gate-level Verilog file
        """
        # Check if YoWASP Yosys is installed
        if shutil.which("yowasp-yosys") is None: 
            raise FileNotFoundError("YoWASP Yosys not found. Please install it via 'pip install yowasp-yosys' for behavioral Verilog support.") 
        
        # Create output filename in the same directory as the input file
        filename = filename.replace('\\', '/')
        output_dir = os.path.dirname(filename)
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_file = os.path.join(output_dir, f"{base_name}_synth.v").replace('\\', '/')
        
        # Create Yosys script file
        script_file = os.path.join(output_dir, f"{base_name}_synth.ys")
        with open(script_file, 'w') as f:
            f.write(f"# Yosys synthesis script\n")
            f.write(f"read_verilog {filename}\n")
            f.write(f"hierarchy -check -top {base_name}\n")  # Assuming module name matches file name
            f.write(f"proc; opt; fsm; opt\n")
            f.write(f"memory; opt\n")
            f.write(f"techmap; opt\n")
            f.write(f"write_verilog -noattr {output_file}\n")
        
        # Run Yosys synthesis
        result = subprocess.run(
            ["yowasp-yosys", "-q", "-T", script_file],  
            check=True, 
            capture_output=True, 
            text=True
        )
        
        # Remove temporary script file
        os.remove(script_file)
        
        # Check if output file was created
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Synthesis failed: {result.stderr}")
            
        return output_file
    
    def _parse_verilog_file(self, filename: str) -> None:
        """Parse a Verilog file into an AST using Pyverilog."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.ast, _ = parse([filename])
        self._extract_modules()
    
    def _extract_modules(self) -> None:
        """Extract all module definitions from the AST and their inputs/outputs."""
        for item in self.ast.description.definitions:
            if isinstance(item, ModuleDef):
                self.modules[item.name] = item
                
                input_wires = set()
                output_wires = set()
                
                if hasattr(item, 'portlist') and item.portlist:
                    if hasattr(item.portlist, 'ports'):
                        for port in item.portlist.ports:
                            if hasattr(port, 'first'):
                                io_obj = port.first
                                
                                name = None
                                if hasattr(io_obj, 'name'):
                                    name = io_obj.name
                                elif hasattr(io_obj, 'right') and hasattr(io_obj.right, 'name'):
                                    name = io_obj.right.name
                                
                                if name is None:
                                    continue
                                
                                width = None
                                if hasattr(io_obj, 'width'):
                                    width = io_obj.width
                                
                                if isinstance(io_obj, Input):
                                    input_wires.add(name)
                                    
                                    if width and hasattr(width, 'msb') and hasattr(width, 'lsb'):
                                        try:
                                            msb = int(width.msb.value) if hasattr(width.msb, 'value') else 0
                                            lsb = int(width.lsb.value) if hasattr(width.lsb, 'value') else 0
                                            bit_width = abs(msb - lsb) + 1
                                            
                                            for i in range(bit_width):
                                                bit_name = f"{name}[{i}]"
                                                input_wires.add(bit_name)
                                        except (ValueError, TypeError, AttributeError):
                                            pass
                                
                                elif isinstance(io_obj, Output):
                                    output_wires.add(name)
                                    
                                    if width and hasattr(width, 'msb') and hasattr(width, 'lsb'):
                                        try:
                                            msb = int(width.msb.value) if hasattr(width.msb, 'value') else 0
                                            lsb = int(width.lsb.value) if hasattr(width.lsb, 'value') else 0
                                            bit_width = abs(msb - lsb) + 1
                                            
                                            for i in range(bit_width):
                                                bit_name = f"{name}[{i}]"
                                                output_wires.add(bit_name)
                                        except (ValueError, TypeError, AttributeError):
                                            pass
                
                self.module_inputs[item.name] = input_wires
                self.module_outputs[item.name] = output_wires
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        for file in self.temp_files:
            if os.path.exists(file):
                os.remove(file)

class ModuleFlattener:
    """Flattens module hierarchies into a single-level netlist."""
    def __init__(self, parser: VerilogParser):
        self.parser = parser
        self.flattened_gates: Dict[str, GateInfo] = {}
        self.wire_map: Dict[str, str] = {}
        self.input_wires: Set[str] = set()
        self.output_wires: Set[str] = set()
        self.gate_counter = 0
        self.current_module: Optional[str] = None
        # Store initial values for registers
        self.initial_states: Dict[str, Any] = {}
        # Store reset conditions for analysis
        self.reset_conditions: Dict[str, List[tuple]] = {}
    
    def flatten(self) -> None:
        """Flatten all modules into a single netlist."""
        for module_name, module_def in self.parser.modules.items():
            self.current_module = module_name
            
            if module_name in self.parser.module_inputs:
                self.input_wires.update(self.parser.module_inputs[module_name])
            if module_name in self.parser.module_outputs:
                self.output_wires.update(self.parser.module_outputs[module_name])
            
            self._process_module(module_def)
            
        # Process initial states after all modules are processed
        self._process_initial_states()
    
    def _process_module(self, module_def: ModuleDef) -> None:
        """Process a module definition and its instances."""
        for item in module_def.items:
            if isinstance(item, InstanceList):
                self._process_instance_list(item)
            elif isinstance(item, Decl):
                self._process_declaration(item)
            elif isinstance(item, Always):
                # Process always blocks for reset conditions and initial states
                self._process_always_block(item)
            elif isinstance(item, Initial):
                # Process initial blocks for initial states
                self._process_initial_block(item)
                
    def _process_declaration(self, decl: Decl) -> None:
        """Process wire, input, and output declarations."""
        if not isinstance(decl, Decl):
            return
            
        for item in decl.list:
            # Track input and output wires
            if isinstance(item, Input):
                self._add_to_wire_map(item)
                if hasattr(item, 'name'):
                    self.input_wires.add(item.name)
                    if hasattr(item, 'width') and item.width is not None:
                        base_name = item.name.split('[')[0] if '[' in item.name else item.name
                        self.input_wires.add(base_name)
                elif hasattr(item, 'names'):
                    for name in item.names:
                        if isinstance(name, Identifier):
                            self.input_wires.add(name.name)
                            if hasattr(item, 'width') and item.width is not None:
                                base_name = name.name.split('[')[0] if '[' in name.name else name.name
                                self.input_wires.add(base_name)
            elif isinstance(item, Output):
                self._add_to_wire_map(item)
                if hasattr(item, 'name'):
                    self.output_wires.add(item.name)
                    if hasattr(item, 'width') and item.width is not None:
                        base_name = item.name.split('[')[0] if '[' in item.name else item.name
                        self.output_wires.add(base_name)
                elif hasattr(item, 'names'):
                    for name in item.names:
                        if isinstance(name, Identifier):
                            self.output_wires.add(name.name)
                            if hasattr(item, 'width') and item.width is not None:
                                base_name = name.name.split('[')[0] if '[' in name.name else name.name
                                self.output_wires.add(base_name)
            elif isinstance(item, Wire):
                self._add_to_wire_map(item)
    
    def _add_to_wire_map(self, item: Union[Input, Output, Wire]) -> None:
        """Add wire to wire_map with proper bit handling."""
        if hasattr(item, 'width') and item.width is not None:
            try:
                # Convert string values to integers
                msb = int(item.width.msb.value) if isinstance(item.width.msb, IntConst) else int(item.width.msb)
                lsb = int(item.width.lsb.value) if isinstance(item.width.lsb, IntConst) else int(item.width.lsb)
                width = abs(msb - lsb) + 1
                
                # Create individual wire entries for each bit
                if hasattr(item, 'name'):
                    base_name = item.name
                    for i in range(width):
                        bit_name = f"{base_name}[{i}]"
                        self.wire_map[bit_name] = bit_name
                elif hasattr(item, 'names'):
                    for name in item.names:
                        if isinstance(name, Identifier):
                            base_name = name.name
                            for i in range(width):
                                bit_name = f"{base_name}[{i}]"
                                self.wire_map[bit_name] = bit_name
            except (ValueError, TypeError) as e:
                warnings.warn(f"Could not parse width for {getattr(item, 'name', 'unknown')}: {e}")
                # Default to single bit if width parsing fails
                self._add_single_bit_wire(item)
        else:
            # Handle single-bit wires
            self._add_single_bit_wire(item)
    
    def _add_single_bit_wire(self, item: Union[Input, Output, Wire]) -> None:
        """Add single-bit wire to wire_map."""
        if hasattr(item, 'name'):
            self.wire_map[item.name] = item.name
        elif hasattr(item, 'names'):
            for name in item.names:
                if isinstance(name, Identifier):
                    self.wire_map[name.name] = name.name
    
    def _process_instance_list(self, instance_list: InstanceList) -> None:
        """Process a list of module instances."""
        for instance in instance_list.instances:
            # Map Yosys cell types to standard gate types
            instance_module = instance.module
            if instance_module in YOSYS_GATE_MAP:
                instance_module = YOSYS_GATE_MAP[instance_module]
                
            if instance_module in self.parser.modules:
                # Recursively process submodule
                self._process_module(self.parser.modules[instance_module])
            else:
                # Process primitive gate
                self._process_gate_instance(instance, instance_module)
    
    def _process_gate_instance(self, instance: Instance, gate_type: str) -> None:
        """Process a primitive gate instance."""
        gate_id = f"g{self.gate_counter}"
        self.gate_counter += 1
        
        # Get instance name with fallback
        instance_name = self._get_instance_name(instance)
        
        # Extract port connections using port mapping for standard gates
        inputs, output, clock = self._map_gate_ports(instance, gate_type)
        
        # Determine gate category
        category = "memory" if gate_type in ["dff", "dffe", "dffs", "dffr", "dffrse", "latch"] else "logic"
        
        # Create gate info with module origin
        gate_info = GateInfo(
            id=gate_id,
            type=gate_type,
            instance_name=instance_name,
            inputs=inputs,
            output=output,
            category=category,
            origin_module=self.current_module,
            clock=clock
        )
        
        self.flattened_gates[gate_id] = gate_info
    
    def _get_instance_name(self, instance: Instance) -> str:
        """Get instance name with fallbacks."""
        if hasattr(instance, 'name'):
            return instance.name
        # If no name, try to get it from the portlist
        elif instance.portlist and hasattr(instance.portlist[0], 'portname'):
            return instance.portlist[0].portname
        # If still no name, generate one based on gate type and counter
        return f"{instance.module}_{self.gate_counter}"
    
    def _map_gate_ports(self, instance: Instance, gate_type: str) -> Tuple[List[str], str, Optional[str]]:
        """
        Map gate ports based on standard port names.
        
        Identifies inputs, outputs, and clock signals for gates based on port mappings
        defined in GATE_PORT_MAP or through intelligent detection.
        
        Args:
            instance: The instance to map ports for
            gate_type: The type of gate
            
        Returns:
            A tuple of (inputs, output, clock) where inputs is a list of input wire names,
            output is the output wire name, and clock is the clock signal name (if applicable)
        """
        inputs = []
        output = None
        clock = None
        
        # Handle special case for Yosys-generated cells
        if gate_type in GATE_PORT_MAP:
            port_map = {}
            # Map each port to its name based on the standard mapping
            for port in instance.portlist:
                if not hasattr(port, 'portname') or not hasattr(port, 'argname'):
                    continue
                port_map[port.portname] = self._resolve_wire(port.argname)
            
            # Fill in inputs and outputs based on standard mapping
            std_map = GATE_PORT_MAP[gate_type]
            
            # Get inputs
            for input_port in std_map['inputs']:
                if input_port in port_map:
                    inputs.append(port_map[input_port])
            
            # Get output
            for output_port in std_map['output']:
                if output_port in port_map:
                    output = port_map[output_port]
                    break  # Just use first output
            
            # Get clock signal for sequential gates
            if 'clock' in std_map:
                for clock_port in std_map['clock']:
                    if clock_port in port_map:
                        clock = port_map[clock_port]
                        break
        else:
            # Fallback for non-standard gates
            for port in instance.portlist:
                if not hasattr(port, 'portname') or not hasattr(port, 'argname'):
                    continue
                    
                # Get the actual wire name
                wire_name = self._resolve_wire(port.argname)
                
                # Determine if this is an input or output port
                if port.portname in ['Y', 'Q', 'out', 'OUT', 'Output', 'output', 'dout', 'DOUT']:  # Common output ports
                    output = wire_name
                else:  # Assume input port
                    inputs.append(wire_name)
                    
                    # Check if this is a clock signal by port name
                    port_name_lower = port.portname.lower()
                    if port_name_lower in [name.lower() for name in CLOCK_SIGNAL_NAMES]:
                        clock = wire_name
                    
                    # Also check if wire name suggests it's a clock
                    wire_name_lower = wire_name.lower()
                    if clock is None and any(clock_name in wire_name_lower for clock_name in [name.lower() for name in CLOCK_SIGNAL_NAMES]):
                        clock = wire_name
        
        # If no output was found, try to infer from the first port
        if output is None and instance.portlist:
            output = self._resolve_wire(instance.portlist[0].argname)
            inputs = [self._resolve_wire(p.argname) for p in instance.portlist[1:]]
            
        return inputs, output, clock
    
    def _resolve_wire(self, wire: Any) -> str:
        """
        Resolve a wire reference to its actual name.
        
        Handles various wire expressions including:
        - Simple identifiers
        - Bit selects (wire[0])
        - Part selects (wire[7:0])
        - Constants
        - Concatenations ({wire1, wire2})
        - Simple arithmetic expressions (wire + 1, wire - 1)
        
        Args:
            wire: The wire expression to resolve
            
        Returns:
            The resolved wire name as a string
            
        Raises:
            ValueError: If a complex or unsupported wire expression is encountered
        """
        try:
            if isinstance(wire, Identifier):
                return wire.name
                
            elif isinstance(wire, Partselect):
                # Part select: wire[msb:lsb]
                if not hasattr(wire, 'var') or not hasattr(wire, 'msb') or not hasattr(wire, 'lsb'):
                    raise ValueError(f"Invalid part select: {wire}")
                    
                var_name = wire.var.name if isinstance(wire.var, Identifier) else str(wire.var)
                msb_val = self._resolve_constant(wire.msb)
                lsb_val = self._resolve_constant(wire.lsb)
                
                return f"{var_name}[{msb_val}:{lsb_val}]"
                
            elif isinstance(wire, Pointer):
                # Bit select: wire[index]
                if not hasattr(wire, 'var') or not hasattr(wire, 'ptr'):
                    raise ValueError(f"Invalid pointer: {wire}")
                    
                var_name = wire.var.name if isinstance(wire.var, Identifier) else str(wire.var)
                index_val = self._resolve_constant(wire.ptr)
                
                return f"{var_name}[{index_val}]"
                
            elif isinstance(wire, IntConst):
                # Integer constant
                return str(wire.value)
                
            elif isinstance(wire, FloatConst):
                # Float constant
                return str(wire.value)
                
            elif isinstance(wire, Concat):
                # Concatenation: {wire1, wire2, ...}
                if not hasattr(wire, 'list'):
                    raise ValueError(f"Invalid concatenation: {wire}")
                    
                concat_items = []
                for item in wire.list:
                    concat_items.append(self._resolve_wire(item))
                
                return "{" + ", ".join(concat_items) + "}"
                
            elif isinstance(wire, Operator):
                # Simple arithmetic: wire + 1, wire - 1
                if not hasattr(wire, 'left') or not hasattr(wire, 'right') or not hasattr(wire, 'operator'):
                    raise ValueError(f"Invalid operator: {wire}")
                    
                # Only support very simple expressions with constants
                left_val = self._resolve_wire(wire.left)
                right_val = self._resolve_wire(wire.right)
                
                # Check if one side is a constant
                if (wire.operator in ['+', '-', '*', '/']) and \
                   (self._is_constant_string(left_val) or self._is_constant_string(right_val)):
                    return f"({left_val} {wire.operator} {right_val})"
                else:
                    raise ValueError(
                        f"Complex wire expression not supported: {wire}. Only simple arithmetic with constants is allowed. "
                        f"Please use gate-level netlist or synthesize behavioral Verilog."
                    )
                    
            elif isinstance(wire, Cond):
                # Conditional expression: condition ? true_expr : false_expr
                raise ValueError(
                    f"Conditional wire expression not supported: {wire}. "
                    f"Please use gate-level netlist or synthesize behavioral Verilog."
                )
                
            elif isinstance(wire, FunctionCall):
                # Function call: func(arg1, arg2, ...)
                raise ValueError(
                    f"Function call in wire expression not supported: {wire}. "
                    f"Please use gate-level netlist or synthesize behavioral Verilog."
                )
                
            elif isinstance(wire, str):
                return wire
                
            else:
                return str(wire)
                
        except Exception as e:
            raise ValueError(
                f"Error processing wire expression: {wire}. {str(e)}\n"
                f"Please use gate-level netlist or synthesize behavioral Verilog for complex expressions."
            )
            
    def _resolve_constant(self, node: Any) -> Union[int, str]:
        """
        Resolve a constant value from a node.
        
        Args:
            node: The node to resolve
            
        Returns:
            The resolved constant value as int or string
            
        Raises:
            ValueError: If the node is not a constant
        """
        if isinstance(node, IntConst):
            return node.value
        elif isinstance(node, Identifier):
            # For parameters that might be defined elsewhere
            return node.name
        elif hasattr(node, 'value'):
            return node.value
        else:
            return str(node)
            
    def _is_constant_string(self, value: str) -> bool:
        """
        Check if a string represents a constant value.
        
        Args:
            value: The string to check
            
        Returns:
            True if the string represents a constant, False otherwise
        """
        try:
            int(value)
            return True
        except ValueError:
            # Check for Verilog-style constants: 8'h00, 4'b0101, etc.
            if "'" in value and (
                value.split("'")[1].startswith('h') or 
                value.split("'")[1].startswith('b') or
                value.split("'")[1].startswith('d') or
                value.split("'")[1].startswith('o')
            ):
                return True
            return False
    
    def _process_always_block(self, always_block: Always) -> None:
        """
        Process always blocks to extract reset conditions and initial states for memory elements.
        
        Args:
            always_block: The always block to process
        """
        # Check if this is a sequential block (has posedge/negedge)
        if not hasattr(always_block, 'sens_list') or not always_block.sens_list:
            return
            
        # Look for reset conditions in sensitivity list
        reset_signal = None
        
        if isinstance(always_block.sens_list, list):
            for sens in always_block.sens_list:
                if isinstance(sens, Sens) and hasattr(sens, 'sig') and hasattr(sens, 'type'):
                    if sens.type == 'posedge' and self._is_reset_signal(sens.sig.name):
                        reset_signal = sens.sig.name
                        break
        elif hasattr(always_block.sens_list, 'list') and isinstance(always_block.sens_list.list, list):
            for sens in always_block.sens_list.list:
                if isinstance(sens, Sens) and hasattr(sens, 'sig') and hasattr(sens, 'type'):
                    if sens.type == 'posedge' and self._is_reset_signal(sens.sig.name):
                        reset_signal = sens.sig.name
                        break
        elif isinstance(always_block.sens_list, Sens):
            sens = always_block.sens_list
            if hasattr(sens, 'sig') and hasattr(sens, 'type'):
                if sens.type == 'posedge' and self._is_reset_signal(sens.sig.name):
                    reset_signal = sens.sig.name
        
        # If no reset signal found, return
        if not reset_signal:
            return
            
        # Process the statement for reset conditions
        if hasattr(always_block, 'statement') and always_block.statement:
            self._extract_reset_conditions(always_block.statement, reset_signal)
    
    def _is_reset_signal(self, signal_name: str) -> bool:
        """
        Check if a signal name is likely a reset signal.
        
        Args:
            signal_name: Name of the signal to check
            
        Returns:
            True if the signal is likely a reset signal, False otherwise
        """
        reset_keywords = ['rst', 'reset', 'clear', 'clr', 'arst', 'aresetn', 'resetn']
        signal_lower = signal_name.lower()
        
        for keyword in reset_keywords:
            if keyword in signal_lower:
                return True
                
        return False
    
    def _extract_reset_conditions(self, statement, reset_signal: str) -> None:
        """
        Extract reset conditions from a statement.
        
        Args:
            statement: The statement to process
            reset_signal: The identified reset signal
        """
        # Handle If statements for reset conditions
        if isinstance(statement, If):
            # Check if condition involves the reset signal
            cond = statement.cond
            if self._condition_has_signal(cond, reset_signal):
                # Process the true path for reset assignments
                if hasattr(statement, 'true_statement') and statement.true_statement:
                    self._process_reset_assignments(statement.true_statement, reset_signal)
        
        # Process other types of statements as needed
        elif isinstance(statement, Block):
            if hasattr(statement, 'statements') and statement.statements:
                for stmt in statement.statements:
                    self._extract_reset_conditions(stmt, reset_signal)
    
    def _condition_has_signal(self, condition, signal_name: str) -> bool:
        """
        Check if a condition involves a specific signal.
        
        Args:
            condition: The condition to check
            signal_name: Name of the signal to look for
            
        Returns:
            True if the condition involves the signal, False otherwise
        """
        # Handle simple identifier
        if isinstance(condition, Identifier) and condition.name == signal_name:
            return True
            
        # Handle complex expressions
        if hasattr(condition, 'left') and condition.left:
            if isinstance(condition.left, Identifier) and condition.left.name == signal_name:
                return True
            elif hasattr(condition.left, 'var') and condition.left.var and \
                 isinstance(condition.left.var, Identifier) and condition.left.var.name == signal_name:
                return True
        
        if hasattr(condition, 'right') and condition.right:
            if isinstance(condition.right, Identifier) and condition.right.name == signal_name:
                return True
            elif hasattr(condition.right, 'var') and condition.right.var and \
                 isinstance(condition.right.var, Identifier) and condition.right.var.name == signal_name:
                return True
        
        return False
    
    def _process_reset_assignments(self, statement, reset_signal: str) -> None:
        """
        Process assignments in reset condition blocks.
        
        Args:
            statement: The statement to process
            reset_signal: The identified reset signal
        """
        if isinstance(statement, NonblockingSubstitution):
            # Handle register assignments during reset
            if hasattr(statement, 'left') and statement.left and hasattr(statement, 'right') and statement.right:
                reg_name = self._get_variable_name(statement.left)
                initial_value = self._get_constant_value(statement.right)
                
                if reg_name and initial_value is not None:
                    # Store the reset condition
                    if reg_name not in self.reset_conditions:
                        self.reset_conditions[reg_name] = []
                    self.reset_conditions[reg_name].append((reset_signal, initial_value))
        
        # Process block statements
        elif isinstance(statement, Block):
            if hasattr(statement, 'statements') and statement.statements:
                for stmt in statement.statements:
                    self._process_reset_assignments(stmt, reset_signal)
    
    def _get_variable_name(self, node) -> Optional[str]:
        """
        Extract variable name from an AST node.
        
        Args:
            node: The AST node to process
            
        Returns:
            The variable name if found, None otherwise
        """
        if isinstance(node, Identifier):
            return node.name
        elif hasattr(node, 'var') and node.var and isinstance(node.var, Identifier):
            # Handle indexed variables (e.g., reg[0])
            var_name = node.var.name
            
            # Handle bit select
            if hasattr(node, 'ptr') and node.ptr:
                index = self._get_constant_value(node.ptr)
                if index is not None:
                    return f"{var_name}[{index}]"
            
            # Handle part select
            if hasattr(node, 'msb') and node.msb and hasattr(node, 'lsb') and node.lsb:
                msb = self._get_constant_value(node.msb)
                lsb = self._get_constant_value(node.lsb)
                if msb is not None and lsb is not None:
                    return f"{var_name}[{msb}:{lsb}]"
            
            return var_name
        
        return None
    
    def _get_constant_value(self, node) -> Optional[Any]:
        """
        Extract constant value from an AST node.
        
        Args:
            node: The AST node to process
            
        Returns:
            The constant value if found, None otherwise
        """
        if isinstance(node, IntConst):
            return node.value
        elif isinstance(node, FloatConst):
            return node.value
        elif isinstance(node, StringConst):
            return node.value
        elif isinstance(node, Concat):
            # Handle concatenation (e.g., {1'b0, 1'b1})
            if hasattr(node, 'list') and node.list:
                concat_values = []
                for item in node.list:
                    val = self._get_constant_value(item)
                    if val is not None:
                        concat_values.append(val)
                    else:
                        return None  # If any item is not a constant, return None
                return concat_values
        
        return None
    
    def _process_initial_block(self, initial_block: Initial) -> None:
        """
        Process initial blocks to extract initial values for registers.
        
        Args:
            initial_block: The initial block to process
        """
        if hasattr(initial_block, 'statement') and initial_block.statement:
            self._process_initial_statements(initial_block.statement)
    
    def _process_initial_statements(self, statement) -> None:
        """
        Process statements in initial blocks.
        
        Args:
            statement: The statement to process
        """
        if isinstance(statement, NonblockingSubstitution) or isinstance(statement, BlockingSubstitution):
            # Handle register assignments in initial blocks
            if hasattr(statement, 'left') and statement.left and hasattr(statement, 'right') and statement.right:
                reg_name = self._get_variable_name(statement.left)
                initial_value = self._get_constant_value(statement.right)
                
                if reg_name and initial_value is not None:
                    # Store the initial state
                    self.initial_states[reg_name] = initial_value
        
        # Process block statements
        elif isinstance(statement, Block):
            if hasattr(statement, 'statements') and statement.statements:
                for stmt in statement.statements:
                    self._process_initial_statements(stmt)
    
    def _process_initial_states(self) -> None:
        """
        Process and apply initial states to memory gates after all modules have been processed.
        """
        for gate_id, gate_info in self.flattened_gates.items():
            # Only process memory elements
            if gate_info.category != 'memory':
                continue
                
            # Check if we have a reset condition for this gate's output
            if gate_info.output in self.reset_conditions:
                # Use the first reset condition (prioritize active high resets)
                reset_cond = self.reset_conditions[gate_info.output][0]
                gate_info.initial_state = reset_cond[1]
            # Check if we have an initial state for this gate's output
            elif gate_info.output in self.initial_states:
                gate_info.initial_state = self.initial_states[gate_info.output]
            else:
                # Default initial state is 0 for memory elements
                gate_info.initial_state = 0

class DAGBuilder:
    """Builds a DAG representation of the flattened circuit."""
    def __init__(self, flattener: ModuleFlattener, fault_model: Optional[FaultModel] = None):
        self.flattener = flattener
        self.fault_model = fault_model
        self.graph = nx.DiGraph()
        self.fault_points: Set[str] = set()
        self.critical_gates: Set[str] = set()
        self.unrolled_graph: Optional[nx.DiGraph] = None  # Store unrolled circuit graph
        self.memory_gates: Dict[str, Dict] = {}  # Track memory gates for unrolling
    
    def build(self) -> None:
        """Build the DAG from flattened gates."""
        # Add nodes for all gates
        for gate_id, gate_info in self.flattener.flattened_gates.items():
            self.graph.add_node(gate_id, **gate_info.__dict__)
            
            # Track memory gates for later unrolling
            if gate_info.category == 'memory':
                self.memory_gates[gate_id] = {
                    'inputs': gate_info.inputs,
                    'output': gate_info.output,
                    'clock': gate_info.clock,
                    'initial_state': gate_info.initial_state
                }
            
            # Add nodes for inputs that don't have driving gates
            for input_wire in gate_info.inputs:
                driving_gate_found = False
                for other_id, other_gate in self.flattener.flattened_gates.items():
                    if other_gate.output == input_wire:
                        driving_gate_found = True
                        break
                
                # If no driving gate, add as input node
                if not driving_gate_found:
                    if input_wire not in self.graph:
                        is_external = input_wire in self.flattener.input_wires
                        self.graph.add_node(input_wire, 
                                          type="input", 
                                          instance_name=input_wire,
                                          inputs=[],
                                          output=input_wire,
                                          category="external" if is_external else "wire",
                                          origin_module=self.flattener.current_module)
            
            # Add edges for all inputs
            for input_wire in gate_info.inputs:
                source_found = False
                # Find gates that drive this input
                for other_id, other_gate in self.flattener.flattened_gates.items():
                    if other_gate.output == input_wire:
                        self.graph.add_edge(other_id, gate_id)
                        source_found = True
                
                # If input has no source gate, connect from input node
                if not source_found and input_wire in self.graph:
                    self.graph.add_edge(input_wire, gate_id)
        
        for input_wire in self.flattener.input_wires:
            if input_wire not in self.graph:
                self.graph.add_node(input_wire,
                                  type="input",
                                  instance_name=input_wire,
                                  inputs=[],
                                  output=input_wire,
                                  category="external",
                                  origin_module=self.flattener.current_module)
                                  
        for output_wire in self.flattener.output_wires:
            if output_wire not in self.graph:
                driving_gate = None
                for gate_id, gate_info in self.flattener.flattened_gates.items():
                    if gate_info.output == output_wire:
                        driving_gate = gate_id
                        break
                
                if driving_gate:
                    if driving_gate not in self.graph:
                        continue 
                        
                    self.graph.add_node(output_wire,
                                      type="output",
                                      instance_name=output_wire,
                                      inputs=[],
                                      output=output_wire,
                                      category="external",
                                      origin_module=self.flattener.current_module)
                    self.graph.add_edge(driving_gate, output_wire)
                else:
                    self.graph.add_node(output_wire,
                                      type="output",
                                      instance_name=output_wire,
                                      inputs=[],
                                      output=output_wire,
                                      category="external",
                                      origin_module=self.flattener.current_module)
        
        def extract_base_names(signal_list):
            result = set()
            for signal in signal_list:
                result.add(signal)
                if '[' in signal:
                    base_name = signal.split('[')[0]
                    result.add(base_name)
            return list(result)
            
        input_signals = extract_base_names(self.flattener.input_wires)
        output_signals = extract_base_names(self.flattener.output_wires)
        
        input_signals = list(set(input_signals))
        output_signals = list(set(output_signals))
        
        # Store external inputs and outputs in graph metadata
        self.graph.graph['inputs'] = input_signals
        self.graph.graph['outputs'] = output_signals
        
        # Identify fault points
        self._identify_fault_points()
        
        # Identify critical gates (those that affect outputs)
        self._identify_critical_gates()
    
    def unroll_circuit(self, max_cycles: int = 1) -> nx.DiGraph:
        """
        Unroll the circuit for multi-cycle analysis.
        
        This method creates a copy of the circuit for each cycle and connects
        memory gates between cycles, allowing for multi-cycle fault analysis.
        
        Args:
            max_cycles: Maximum number of cycles to unroll
            
        Returns:
            The unrolled circuit as a DiGraph
        """
        if max_cycles <= 0:
            raise ValueError("max_cycles must be a positive integer")
            
        # Create a new graph for the unrolled circuit
        unrolled = nx.DiGraph()
        unrolled.graph['inputs'] = self.graph.graph.get('inputs', []).copy()
        unrolled.graph['outputs'] = self.graph.graph.get('outputs', []).copy()
        
        # Map of original node ID to unrolled node IDs for each cycle
        # Format: {original_id: {cycle: unrolled_id}}
        node_map = {}
        
        # Map of memory gate outputs from previous cycles
        # Format: {output_wire: {cycle: node_id}}
        memory_outputs = {}
        
        # Process each cycle
        for cycle in range(max_cycles):
            # Create a copy of all nodes for this cycle
            for node_id in self.graph.nodes():
                original_data = self.graph.nodes[node_id].copy()
                
                # Create a new node ID for this cycle
                cycle_node_id = f"{node_id}_c{cycle}"
                
                # Initialize node map entry if not exists
                if node_id not in node_map:
                    node_map[node_id] = {}
                
                # Map original node to the cycle-specific node
                node_map[node_id][cycle] = cycle_node_id
                
                # Add the node to the unrolled graph
                unrolled.add_node(cycle_node_id, **original_data)
                
                # Track memory gate outputs
                if original_data.get('category') == 'memory':
                    output_wire = original_data.get('output')
                    if output_wire not in memory_outputs:
                        memory_outputs[output_wire] = {}
                    memory_outputs[output_wire][cycle] = cycle_node_id
        
        # Process each cycle again to add edges
        for cycle in range(max_cycles):
            for node_id in self.graph.nodes():
                original_data = self.graph.nodes[node_id]
                cycle_node_id = node_map[node_id][cycle]
                
                # Get input wires for this node
                input_wires = original_data.get('inputs', [])
                
                for input_wire in input_wires:
                    # Check if this input comes from a memory gate in a previous cycle
                    if cycle > 0 and input_wire in memory_outputs and (cycle-1) in memory_outputs[input_wire]:
                        # Connect to the memory gate from the previous cycle
                        prev_memory_node = memory_outputs[input_wire][cycle-1]
                        unrolled.add_edge(prev_memory_node, cycle_node_id)
                    else:
                        # Find nodes in the current cycle that drive this input
                        for src_id, src_data in self.graph.nodes(data=True):
                            if src_data.get('output') == input_wire:
                                src_cycle_id = node_map[src_id][cycle]
                                unrolled.add_edge(src_cycle_id, cycle_node_id)
        
        # Set initial state for memory gates in cycle 0
        for gate_id, gate_data in self.memory_gates.items():
            cycle0_gate_id = node_map[gate_id][0]
            # Set the initial state attribute
            if 'initial_state' in gate_data and gate_data['initial_state'] is not None:
                unrolled.nodes[cycle0_gate_id]['initial_state'] = gate_data['initial_state']
        
        # Store the unrolled graph
        self.unrolled_graph = unrolled
        
        return unrolled
    
    def get_unrolled_graph(self) -> Optional[nx.DiGraph]:
        """
        Get the unrolled circuit graph if available.
        
        Returns:
            The unrolled circuit graph, or None if not created yet
        """
        return self.unrolled_graph
    
    def _identify_fault_points(self) -> None:
        """
        Identify potential fault injection points based on the fault model.
        
        If a fault model is specified, respects gate type restrictions
        and blacklist.
        """
        self.fault_points.clear()
        
        for node in self.graph.nodes():
            if node not in self.graph:
                continue
                
            node_data = self.graph.nodes[node]
            
            # Skip nodes that aren't gates
            if 'category' not in node_data:
                continue
                
            category = node_data['category']
            
            # Check if this node should be considered for fault injection
            if category in ['logic', 'memory']:
                # If fault model specified, apply restrictions
                if self.fault_model is not None:
                    # Check gate type restrictions
                    allowed_types = []
                    if self.fault_model.gate_types == GateType.LOGIC or self.fault_model.gate_types == GateType.BOTH:
                        allowed_types.append('logic')
                    if self.fault_model.gate_types == GateType.MEMORY or self.fault_model.gate_types == GateType.BOTH:
                        allowed_types.append('memory')
                        
                    if category not in allowed_types:
                        continue
                        
                    # Check blacklist
                    if hasattr(self.fault_model, 'blacklist') and node in self.fault_model.blacklist:
                        continue
                
                self.fault_points.add(node)
    
    def _identify_critical_gates(self) -> None:
        """
        Identify gates that affect outputs (critical gates).
        This helps reduce the number of fault points to consider.
        """
        self.critical_gates.clear()
        
        # Get all output nodes
        outputs = self.graph.graph.get('outputs', [])
        
        # For each output, find all ancestors
        for output in outputs:
            # Find all nodes that drive this output
            output_drivers = []
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]
                if 'output' in node_data and node_data['output'] == output:
                    output_drivers.append(node)
            
            # For each driver, add it and all its ancestors to critical gates
            for driver in output_drivers:
                self.critical_gates.add(driver)
                ancestors = nx.ancestors(self.graph, driver)
                self.critical_gates.update(ancestors)
    
    def reduce_fault_points(self) -> None:
        """
        Reduce fault points to only those that are critical.
        Critical gates are those that affect outputs.
        """
        if not self.critical_gates:
            self._identify_critical_gates()
            
        self.fault_points = self.fault_points.intersection(self.critical_gates)
    
    def to_json(self) -> str:
        """Convert the DAG to JSON format."""
        graph_data = {
            "nodes": [],
            "edges": [],
            "fault_points": list(self.fault_points),
            "critical_gates": list(self.critical_gates),
            "module_hierarchy": {},  # Track module hierarchy
            "inputs": self.graph.graph.get('inputs', []),
            "outputs": self.graph.graph.get('outputs', [])
        }
        
        # Add nodes
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            node_entry = {
                "id": node,
            }
            
            # Add all node attributes that exist
            for attr in ["type", "instance_name", "inputs", "output", "category", 
                         "origin_module", "bit_width", "bit_range", "clock"]:
                if attr in node_data:
                    node_entry[attr] = node_data[attr]
            
            graph_data["nodes"].append(node_entry)
            
            # Track module hierarchy
            origin_module = node_data.get("origin_module", "unknown")
            if origin_module not in graph_data["module_hierarchy"]:
                graph_data["module_hierarchy"][origin_module] = {
                    "gate_count": 0,
                    "fault_points": 0
                }
            graph_data["module_hierarchy"][origin_module]["gate_count"] += 1
            if node in self.fault_points:
                graph_data["module_hierarchy"][origin_module]["fault_points"] += 1
        
        # Add edges
        for edge in self.graph.edges():
            graph_data["edges"].append({
                "source": edge[0],
                "target": edge[1]
            })
        
        return json.dumps(graph_data, indent=2)
    
    def visualize(self, filename: str = "circuit") -> None:
        """Create a Graphviz visualization of the circuit."""
        try:
            dot = graphviz.Digraph(comment='Circuit DAG')
            dot.attr(rankdir='LR')
            
            # Add nodes
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]
                
                # Determine node color
                color = 'black'
                if node in self.fault_points:
                    color = 'red'  # Fault points
                elif node in self.critical_gates:
                    color = 'blue'  # Critical gates
                
                # Create label with relevant information
                label_parts = []
                for attr in ["type", "instance_name", "category", "origin_module"]:
                    if attr in node_data and node_data[attr]:
                        label_parts.append(f"{attr}: {node_data[attr]}")
                
                label = f"{node}\n" + "\n".join(label_parts)
                
                # Add node with label and color
                shape = "box" if node_data.get("category") in ["logic", "memory"] else "ellipse"
                dot.node(node, label, color=color, shape=shape)
            
            # Add edges
            for edge in self.graph.edges():
                dot.edge(edge[0], edge[1])
            
            # Render the graph
            dot.render(filename, view=False, format='png')
        except Exception as e:
            warnings.warn(f"Could not generate visualization: {e}\n"
                         f"Please install Graphviz and add it to your system PATH")
            warnings.warn(f"You can still use the JSON output for analysis")

def parse_verilog_to_dag(filename: str, 
                        fault_model: Optional[FaultModel] = None,
                        use_synthesis: bool = True,
                        reduce_fault_points: bool = True,
                        unroll_cycles: int = 1,
                        custom_clock_names: Optional[List[str]] = None) -> DAGBuilder:
    """
    Parse a Verilog file and return a DAG representation.
    
    Args:
        filename: Path to Verilog file
        fault_model: Optional fault model to restrict fault points
        use_synthesis: Whether to attempt synthesis for behavioral Verilog
        reduce_fault_points: Whether to reduce fault points to critical gates
        unroll_cycles: Number of cycles to unroll the circuit for multi-cycle analysis
        custom_clock_names: Optional list of custom clock signal names to recognize
        
    Returns:
        DAGBuilder instance with parsed circuit
    """
    parser = VerilogParser()
    
    # Add custom clock signal names if provided
    global CLOCK_SIGNAL_NAMES
    if custom_clock_names:
        CLOCK_SIGNAL_NAMES.extend(custom_clock_names)
    
    try:
        parser.parse_file(filename, use_synthesis=use_synthesis)
        
        flattener = ModuleFlattener(parser)
        flattener.flatten()
        
        dag_builder = DAGBuilder(flattener, fault_model)
        dag_builder.build()
        
        # Reduce fault points if requested
        if reduce_fault_points:
            dag_builder.reduce_fault_points()
        
        # Unroll the circuit for multi-cycle analysis if more than 1 cycle
        if unroll_cycles > 1:
            dag_builder.unroll_circuit(unroll_cycles)
        
        return dag_builder
    finally:
        # Clean up temporary files
        parser.cleanup()