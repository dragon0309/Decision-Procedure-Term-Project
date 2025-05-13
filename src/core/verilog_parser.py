from pyverilog.vparser.parser import parse
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.vparser.ast import *
import networkx as nx
import json
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
import graphviz
from .gadget_transformer import GadgetTransformer
from .cnf_encoder import CNFEncoder

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

class VerilogParser:
    """Parses Verilog files into an AST using Pyverilog."""
    def __init__(self):
        self.ast = None
        self.modules = {}
        
    def parse_file(self, filename: str) -> None:
        """Parse a Verilog file into an AST."""
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.ast, _ = parse([filename])
        self._extract_modules()
    
    def _extract_modules(self) -> None:
        """Extract all module definitions from the AST."""
        for item in self.ast.description.definitions:
            if isinstance(item, ModuleDef):
                self.modules[item.name] = item

class ModuleFlattener:
    """Flattens module hierarchies into a single-level netlist."""
    def __init__(self, parser: VerilogParser):
        self.parser = parser
        self.flattened_gates: Dict[str, GateInfo] = {}
        self.wire_map: Dict[str, str] = {}
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
            if isinstance(item, (Input, Output, Wire)):
                # Handle bit ranges
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
                        print(f"Warning: Could not parse width for {item.name}: {e}")
                        # Default to single bit if width parsing fails
                        if hasattr(item, 'name'):
                            self.wire_map[item.name] = item.name
                        elif hasattr(item, 'names'):
                            for name in item.names:
                                if isinstance(name, Identifier):
                                    self.wire_map[name.name] = name.name
                else:
                    # Handle single-bit wires
                    if hasattr(item, 'name'):
                        self.wire_map[item.name] = item.name
                    elif hasattr(item, 'names'):
                        for name in item.names:
                            if isinstance(name, Identifier):
                                self.wire_map[name.name] = name.name
    
    def _process_instance_list(self, instance_list: InstanceList) -> None:
        """Process a list of module instances."""
        for instance in instance_list.instances:
            if instance.module in self.parser.modules:
                # Recursively process submodule
                self._process_module(self.parser.modules[instance.module])
            else:
                # Process primitive gate
                self._process_gate_instance(instance)
    
    def _process_gate_instance(self, instance: Instance) -> None:
        """Process a primitive gate instance."""
        gate_id = f"g{self.gate_counter}"
        self.gate_counter += 1
        
        # Get instance name with fallback
        # First try to get the instance name from the instance object
        instance_name = None
        if hasattr(instance, 'name'):
            instance_name = instance.name
        # If no name, try to get it from the portlist (some gates use port names as instance names)
        elif instance.portlist and hasattr(instance.portlist[0], 'portname'):
            instance_name = instance.portlist[0].portname
        # If still no name, generate one based on gate type and counter
        if not instance_name:
            instance_name = f"{instance.module}_{self.gate_counter}"
        
        # Extract port connections
        inputs = []
        output = None
        
        # Handle port connections
        for port in instance.portlist:
            if not hasattr(port, 'portname') or not hasattr(port, 'argname'):
                continue
                
            # Get the actual wire name
            wire_name = self._resolve_wire(port.argname)
            
            # Determine if this is an input or output port
            if port.portname in ['Y', 'Q', 'out']:  # Output ports
                output = wire_name
            else:  # Input ports
                inputs.append(wire_name)
        
        # If no output was found, try to infer from the first port
        if output is None and instance.portlist:
            output = self._resolve_wire(instance.portlist[0].argname)
            inputs = [self._resolve_wire(p.argname) for p in instance.portlist[1:]]
        
        # Determine gate category
        category = "memory" if instance.module == "dff" else "logic"
        
        # Create gate info with module origin
        gate_info = GateInfo(
            id=gate_id,
            type=instance.module,
            instance_name=instance_name,
            inputs=inputs,
            output=output,
            category=category,
            origin_module=self.current_module
        )
        
        self.flattened_gates[gate_id] = gate_info
    
    def _resolve_wire(self, wire: Any) -> str:
        """Resolve a wire reference to its actual name."""
        if isinstance(wire, Identifier):
            return wire.name
        elif isinstance(wire, Partselect):
            return f"{wire.var.name}[{wire.msb.value}:{wire.lsb.value}]"
        elif isinstance(wire, Pointer):
            return f"{wire.var.name}[{wire.ptr.value}]"
        elif isinstance(wire, (IntConst, FloatConst)):
            return str(wire.value)
        elif isinstance(wire, str):
            return wire
        return str(wire)

class DAGBuilder:
    """Builds a DAG representation of the flattened circuit."""
    def __init__(self, flattener: ModuleFlattener):
        self.flattener = flattener
        self.graph = nx.DiGraph()
        self.fault_points: Set[str] = set()
    
    def build(self) -> None:
        """Build the DAG from flattened gates."""
        # Add nodes
        for gate_id, gate_info in self.flattener.flattened_gates.items():
            self.graph.add_node(gate_id, **gate_info.__dict__)
            
            # Add edges
            for input_wire in gate_info.inputs:
                # Find gates that drive this input
                for other_id, other_gate in self.flattener.flattened_gates.items():
                    if other_gate.output == input_wire:
                        self.graph.add_edge(other_id, gate_id)
        
        # Identify fault points
        self._identify_fault_points()
    
    def _identify_fault_points(self) -> None:
        """Identify potential fault injection points."""
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            if node_data['category'] in ['logic', 'memory']:
                self.fault_points.add(node)
    
    def to_json(self) -> str:
        """Convert the DAG to JSON format."""
        graph_data = {
            "nodes": [],
            "edges": [],
            "fault_points": list(self.fault_points),
            "module_hierarchy": {}  # Track module hierarchy
        }
        
        # Add nodes
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            graph_data["nodes"].append({
                "id": node,
                "type": node_data["type"],
                "instance_name": node_data["instance_name"],
                "inputs": node_data["inputs"],
                "output": node_data["output"],
                "category": node_data["category"],
                "origin_module": node_data["origin_module"],
                "bit_width": node_data.get("bit_width", 1),
                "bit_range": node_data.get("bit_range")
            })
            
            # Track module hierarchy
            origin_module = node_data["origin_module"]
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
                color = 'red' if node in self.fault_points else 'black'
                label = f"{node_data['type']}\n{node_data['instance_name']}\n({node_data['origin_module']})"
                dot.node(node, label, color=color)
            
            # Add edges
            for edge in self.graph.edges():
                dot.edge(edge[0], edge[1])
            
            dot.render(filename, view=True, format='png')
        except Exception as e:
            print(f"Warning: Could not generate visualization: {e}")
            print("Please install Graphviz and add it to your system PATH")
            print("You can still use the JSON output for analysis")

def parse_verilog_to_dag(filename: str) -> DAGBuilder:
    """Parse a Verilog file and return a DAG representation."""
    parser = VerilogParser()
    parser.parse_file(filename)
    
    flattener = ModuleFlattener(parser)
    flattener.flatten()
    
    dag_builder = DAGBuilder(flattener)
    dag_builder.build()
    
    return dag_builder

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python verilog_parser.py <verilog_file>")
        sys.exit(1)
    
    dag = parse_verilog_to_dag(sys.argv[1])
    print(dag.to_json())
    dag.visualize() 