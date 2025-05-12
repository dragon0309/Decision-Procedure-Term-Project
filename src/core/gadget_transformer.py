from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
import networkx as nx
from .sat_solver import SATSolver, solve_cnf, solve_and_report
from .cnf_encoder import CNFEncoder
from .models import FaultModel, FaultGadget

@dataclass
class FaultGadget:
    """Represents a fault injection gadget for a gate."""
    original_gate_id: str
    control_var: str  # c_i: whether to inject fault
    select_var: Optional[str] = None  # s_i: type of fault (if multiple types supported)
    fault_types: List[str] = None  # List of supported fault types

@dataclass
class FaultModel:
    """Fault injection model parameters."""
    max_faults_per_cycle: int  # ne
    max_cycles: int  # nc
    allowed_fault_types: Set[str]  # T
    allowed_gate_types: Set[str]  # \ell

class GadgetTransformer:
    """Transforms a circuit DAG by adding fault injection gadgets."""
    
    def __init__(self, 
                 dag: nx.DiGraph, 
                 fault_points: Set[str],
                 fault_model: Optional[FaultModel] = None):
        """
        Initialize the transformer.
        
        Args:
            dag: Circuit DAG
            fault_points: Set of gate IDs where faults can be injected
            fault_model: Fault injection model parameters
        """
        self.dag = dag
        self.fault_points = fault_points
        self.fault_model = fault_model or FaultModel(
            max_faults_per_cycle=3,
            max_cycles=1,
            allowed_fault_types={"bit_flip", "set_1", "set_0"},
            allowed_gate_types={"logic", "memory"}
        )
        self.control_vars: Dict[str, List[str]] = {}  # gate -> [control_vars]
        self.time_control_vars: Dict[int, List[str]] = {}  # cycle -> [control_vars]
        self.transformed_dag = nx.DiGraph()
        self.var_counter = 0
    
    def transform(self) -> nx.DiGraph:
        """Transform the circuit by adding fault gadgets."""
        transformed = self.dag.copy()
        
        # Add fault gadgets to eligible gates
        for node in list(transformed.nodes()):
            if self._is_fault_eligible(node):
                self._add_fault_gadget(transformed, node)
        
        self.transformed_dag = transformed
        return transformed
    
    def _is_fault_eligible(self, gate_id: str) -> bool:
        """Check if a gate is eligible for fault injection."""
        if gate_id not in self.fault_points:
            return False
            
        # Check gate type
        gate_data = self.dag.nodes[gate_id]
        gate_type = gate_data["type"]
        
        if gate_type == "dff" and "memory" not in self.fault_model.allowed_gate_types:
            return False
        if gate_type in {"and", "or", "xor", "not"} and "logic" not in self.fault_model.allowed_gate_types:
            return False
            
        return True
    
    def _add_fault_gadget(self, dag: nx.DiGraph, gate_id: str) -> None:
        """Add fault injection gadget to a gate."""
        gate_data = dag.nodes[gate_id]
        
        # Store original output
        if "output" in gate_data:
            gate_data["original_output"] = gate_data["output"]
        
        # Create control variables for each cycle
        control_vars = []
        for t in range(self.fault_model.max_cycles):
            control_var = f"c{gate_id}_t{t}"
            select_var = f"s{gate_id}_t{t}"
            
            # Add to time-indexed control variables
            if t not in self.time_control_vars:
                self.time_control_vars[t] = []
            self.time_control_vars[t].append(control_var)
            
            control_vars.append(control_var)
            
            # Add variables to DAG
            gate_data["fault_control"] = control_var
            gate_data["fault_select"] = select_var
            gate_data["fault_types"] = list(self.fault_model.allowed_fault_types)
        
        self.control_vars[gate_id] = control_vars
    
    def get_control_vars(self) -> Dict[str, List[str]]:
        """Get mapping from gates to their control variables."""
        return self.control_vars
    
    def get_time_control_vars(self) -> Dict[int, List[str]]:
        """Get mapping from cycles to control variables active in that cycle."""
        return self.time_control_vars
    
    def get_fault_gadgets(self) -> Dict[str, FaultGadget]:
        """Get all fault gadgets."""
        fault_gadgets = {}
        for gate_id, control_vars in self.control_vars.items():
            for t, control_var in enumerate(control_vars):
                select_var = f"s{gate_id}_t{t}"
                fault_gadgets[f"{gate_id}_t{t}"] = FaultGadget(
                    original_gate_id=gate_id,
                    control_var=control_var,
                    select_var=select_var,
                    fault_types=list(self.fault_model.allowed_fault_types)
                )
        return fault_gadgets
    
    def to_json(self) -> Dict[str, Any]:
        """Convert the transformed circuit to JSON format."""
        return {
            "fault_points": list(self.fault_points),
            "control_vars": self.control_vars,
            "time_control_vars": self.time_control_vars,
            "fault_gadgets": {
                gate_id: {
                    "control_vars": control_vars,
                    "fault_types": fault_types
                }
                for gate_id, control_vars in self.control_vars.items()
                for t, control_var in enumerate(control_vars)
                for fault_type in self.fault_model.allowed_fault_types
                for fault_types in [list(self.fault_model.allowed_fault_types)]
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