from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
import networkx as nx
from .sat_solver import SATSolver, solve_cnf, solve_and_report
from .cnf_encoder import CNFEncoder

@dataclass
class FaultGadget:
    """Represents a fault injection gadget for a gate."""
    original_gate_id: str
    control_var: str  # c_i: whether to inject fault
    select_var: Optional[str] = None  # s_i: type of fault (if multiple types supported)
    fault_types: List[str] = None  # List of supported fault types

class GadgetTransformer:
    """Transforms a circuit DAG by wrapping fault points with fault gadgets."""
    
    def __init__(self, dag: nx.DiGraph, fault_points: Set[str]):
        self.original_dag = dag
        self.fault_points = fault_points
        self.transformed_dag = nx.DiGraph()
        self.control_vars: Dict[str, str] = {}  # gate_id -> control_var
        self.select_vars: Dict[str, str] = {}   # gate_id -> select_var
        self.fault_gadgets: Dict[str, FaultGadget] = {}
        self.var_counter = 0
    
    def transform(self) -> nx.DiGraph:
        """Transform the circuit by adding fault gadgets."""
        # Copy original DAG structure
        self.transformed_dag = self.original_dag.copy()
        
        # Add fault gadgets for each fault point
        for node in self.fault_points:
            self._add_fault_gadget(node)
        
        return self.transformed_dag
    
    def _add_fault_gadget(self, gate_id: str) -> None:
        """Add a fault gadget for a specific gate."""
        gate_data = self.original_dag.nodes[gate_id]
        
        # Generate control and select variables
        control_var = f"c{self.var_counter}"
        self.var_counter += 1
        select_var = f"s{self.var_counter}"
        self.var_counter += 1
        
        # Store variables
        self.control_vars[gate_id] = control_var
        self.select_vars[gate_id] = select_var
        
        # Create fault gadget
        gadget = FaultGadget(
            original_gate_id=gate_id,
            control_var=control_var,
            select_var=select_var,
            fault_types=["bit_flip", "set_1", "set_0"]  # Example fault types
        )
        self.fault_gadgets[gate_id] = gadget
        
        # Update gate data with fault gadget information
        self.transformed_dag.nodes[gate_id].update({
            "fault_control": control_var,
            "fault_select": select_var,
            "fault_types": gadget.fault_types,
            "original_output": gate_data["output"]
        })
    
    def get_control_vars(self) -> List[str]:
        """Get list of all control variables."""
        return list(self.control_vars.values())
    
    def get_select_vars(self) -> List[str]:
        """Get list of all select variables."""
        return list(self.select_vars.values())
    
    def get_fault_gadgets(self) -> Dict[str, FaultGadget]:
        """Get all fault gadgets."""
        return self.fault_gadgets
    
    def to_json(self) -> Dict[str, Any]:
        """Convert the transformed circuit to JSON format."""
        return {
            "fault_points": list(self.fault_points),
            "control_vars": self.control_vars,
            "select_vars": self.select_vars,
            "fault_gadgets": {
                gate_id: {
                    "control_var": gadget.control_var,
                    "select_var": gadget.select_var,
                    "fault_types": gadget.fault_types
                }
                for gate_id, gadget in self.fault_gadgets.items()
            }
        }

# Example usage
if __name__ == "__main__":
    from verilog_parser import parse_verilog_to_dag
    
    # Parse and transform the circuit
    dag = parse_verilog_to_dag("test_circuit.v")
    transformer = GadgetTransformer(dag.graph, dag.fault_points)
    transformed_dag = transformer.transform()
    
    # Encode to CNF
    encoder = CNFEncoder(transformed_dag)
    encoder.add_constraint_max_faults(3)  # Limit to 3 faults
    cnf = encoder.get_cnf()
    var_map = encoder.get_var_map()
    
    # Solve and analyze
    solve_and_report(cnf, var_map, key_signals=['output', 'fault_control']) 