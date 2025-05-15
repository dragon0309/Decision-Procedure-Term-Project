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

# 為各種閘定義標準連接埠名稱映射
GATE_PORT_MAP = {
    'and': {'inputs': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], 'output': ['Y']},
    'or': {'inputs': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], 'output': ['Y']},
    'xor': {'inputs': ['A', 'B'], 'output': ['Y']},
    'not': {'inputs': ['A'], 'output': ['Y']},
    'nand': {'inputs': ['A', 'B', 'C', 'D'], 'output': ['Y']},
    'nor': {'inputs': ['A', 'B', 'C', 'D'], 'output': ['Y']},
    'buf': {'inputs': ['A'], 'output': ['Y']},
    'mux': {'inputs': ['A', 'B', 'S'], 'output': ['Y']},
    'dff': {'inputs': ['D', 'CLK'], 'output': ['Q', 'QN']},
    'dffe': {'inputs': ['D', 'CLK', 'EN'], 'output': ['Q', 'QN']},
    'dffs': {'inputs': ['D', 'CLK', 'SET'], 'output': ['Q', 'QN']},
    'dffr': {'inputs': ['D', 'CLK', 'RESET'], 'output': ['Q', 'QN']},
    'dffrse': {'inputs': ['D', 'CLK', 'RESET', 'SET', 'EN'], 'output': ['Q', 'QN']},
    'latch': {'inputs': ['D', 'EN'], 'output': ['Q']},
}

# 將 Yosys 特定閘類型映射至標準類型
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

class VerilogParser:
    """Parses Verilog files into an AST using Pyverilog."""
    def __init__(self):
        self.ast = None
        self.modules = {}
        self.temp_files = []
        
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
                              f"Ensure Yosys is installed and in PATH for better behavioral Verilog support.")
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
        # Check if Yosys is installed
        if shutil.which("yosys") is None:
            raise FileNotFoundError("Yosys not found. Please install Yosys for behavioral Verilog support.")
        
        # Create output filename in the same directory as the input file
        output_dir = os.path.dirname(filename)
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_file = os.path.join(output_dir, f"{base_name}_synth.v")
        
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
            ["yosys", "-q", "-T", script_file],
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
        """Extract all module definitions from the AST."""
        for item in self.ast.description.definitions:
            if isinstance(item, ModuleDef):
                self.modules[item.name] = item
                
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
    
    def flatten(self) -> None:
        """Flatten all modules into a single netlist."""
        for module_name, module_def in self.parser.modules.items():
            self.current_module = module_name
            self._process_module(module_def)
    
    def _process_module(self, module_def: ModuleDef) -> None:
        """Process a module definition and its instances."""
        for item in module_def.items:
            if isinstance(item, InstanceList):
                self._process_instance_list(item)
            elif isinstance(item, Decl):
                self._process_declaration(item)
    
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
                elif hasattr(item, 'names'):
                    for name in item.names:
                        if isinstance(name, Identifier):
                            self.input_wires.add(name.name)
            elif isinstance(item, Output):
                self._add_to_wire_map(item)
                if hasattr(item, 'name'):
                    self.output_wires.add(item.name)
                elif hasattr(item, 'names'):
                    for name in item.names:
                        if isinstance(name, Identifier):
                            self.output_wires.add(name.name)
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
        """Map gate ports based on standard port names."""
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
                    # Track clock signal for sequential gates
                    if gate_type in ["dff", "dffe", "dffs", "dffr", "dffrse"] and input_port == "CLK":
                        clock = port_map[input_port]
            
            # Get output
            for output_port in std_map['output']:
                if output_port in port_map:
                    output = port_map[output_port]
                    break  # Just use first output
        else:
            # Fallback for non-standard gates
            for port in instance.portlist:
                if not hasattr(port, 'portname') or not hasattr(port, 'argname'):
                    continue
                    
                # Get the actual wire name
                wire_name = self._resolve_wire(port.argname)
                
                # Determine if this is an input or output port
                if port.portname in ['Y', 'Q', 'out', 'OUT', 'Output']:  # Common output ports
                    output = wire_name
                else:  # Assume input port
                    inputs.append(wire_name)
                    # Check if this is a clock signal
                    if port.portname in ['CLK', 'clk', 'clock', 'CLOCK', 'CK']:
                        clock = wire_name
        
        # If no output was found, try to infer from the first port
        if output is None and instance.portlist:
            output = self._resolve_wire(instance.portlist[0].argname)
            inputs = [self._resolve_wire(p.argname) for p in instance.portlist[1:]]
            
        return inputs, output, clock
    
    def _resolve_wire(self, wire: Any) -> str:
        """
        Resolve a wire reference to its actual name.
        
        Raises:
            ValueError: If a complex wire expression is encountered.
        """
        if isinstance(wire, Identifier):
            return wire.name
        elif isinstance(wire, Partselect):
            return f"{wire.var.name}[{wire.msb.value}:{wire.lsb.value}]"
        elif isinstance(wire, Pointer):
            return f"{wire.var.name}[{wire.ptr.value}]"
        elif isinstance(wire, (IntConst, FloatConst)):
            return str(wire.value)
        elif isinstance(wire, (Operator, Cond)):
            raise ValueError(f"Complex wire expression not supported: {wire}. "
                           f"Please use gate-level netlist or synthesize behavioral Verilog.")
        elif isinstance(wire, str):
            return wire
        return str(wire)

class DAGBuilder:
    """Builds a DAG representation of the flattened circuit."""
    def __init__(self, flattener: ModuleFlattener, fault_model: Optional[FaultModel] = None):
        self.flattener = flattener
        self.fault_model = fault_model
        self.graph = nx.DiGraph()
        self.fault_points: Set[str] = set()
        self.critical_gates: Set[str] = set()
    
    def build(self) -> None:
        """Build the DAG from flattened gates."""
        # Add nodes for all gates
        for gate_id, gate_info in self.flattener.flattened_gates.items():
            self.graph.add_node(gate_id, **gate_info.__dict__)
            
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
        
        # Store external outputs in graph metadata
        self.graph.graph['inputs'] = list(self.flattener.input_wires)
        self.graph.graph['outputs'] = list(self.flattener.output_wires)
        
        # Identify fault points
        self._identify_fault_points()
        
        # Identify critical gates (those that affect outputs)
        self._identify_critical_gates()
    
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
                        reduce_fault_points: bool = True) -> DAGBuilder:
    """
    Parse a Verilog file and return a DAG representation.
    
    Args:
        filename: Path to Verilog file
        fault_model: Optional fault model to restrict fault points
        use_synthesis: Whether to attempt synthesis for behavioral Verilog
        reduce_fault_points: Whether to reduce fault points to critical gates
        
    Returns:
        DAGBuilder instance with parsed circuit
    """
    parser = VerilogParser()
    
    try:
        parser.parse_file(filename, use_synthesis=use_synthesis)
        
        flattener = ModuleFlattener(parser)
        flattener.flatten()
        
        dag_builder = DAGBuilder(flattener, fault_model)
        dag_builder.build()
        
        # Reduce fault points if requested
        if reduce_fault_points:
            dag_builder.reduce_fault_points()
        
        return dag_builder
    finally:
        # Clean up temporary files
        parser.cleanup()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python verilog_parser.py <verilog_file>")
        sys.exit(1)
    
    dag = parse_verilog_to_dag(sys.argv[1])
    print(dag.to_json())
    dag.visualize() 