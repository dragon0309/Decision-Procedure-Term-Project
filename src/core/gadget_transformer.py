from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
import networkx as nx
from .sat_solver import SATSolver, solve_cnf, solve_and_report
from .cnf_encoder import CNFEncoder
from .fault_model import FaultModel, FaultType, GateType

@dataclass
class FaultGadget:
    """Represents a fault injection gadget for a gate."""
    original_gate_id: str
    control_var: str  # c_i: whether to inject fault
    select_var: Optional[str] = None  # s_i: type of fault (if multiple types supported)
    fault_types: List[str] = None  # List of supported fault types

class GadgetTransformer:
    """Transforms a DAG by adding fault gadgets to eligible gates."""
    
    def __init__(self, dag: nx.DiGraph, fault_model: FaultModel):
        """Initialize transformer with DAG and fault model."""
        self.dag = dag
        self.fault_model = fault_model
        self.time_control_vars: Dict[int, List[str]] = {}  # t -> [c1_t, c2_t, ...]
        self.time_select_vars: Dict[int, List[str]] = {}  # t -> [s1_t, s2_t, ...]
        self.gate_fault_vars: Dict[str, Dict[int, str]] = {}  # gate_id -> {t -> f_t}
        self.next_var_id = 1
    
    def _get_next_var(self) -> int:
        """Get next available variable ID."""
        var_id = self.next_var_id
        self.next_var_id += 1
        return var_id
    
    def _is_gate_eligible(self, gate_id: str) -> bool:
        """Check if a gate is eligible for fault injection."""
        gate_data = self.dag.nodes[gate_id]
        return self.fault_model.is_gate_type_allowed(gate_data['category'])
    
    def _add_fault_gadget(self, gate_id: str, t: int) -> None:
        """Add fault gadget for a gate at time t."""
        if gate_id not in self.gate_fault_vars:
            self.gate_fault_vars[gate_id] = {}
        
        # Create control and select variables for this time step
        c_var = f"c_{gate_id}_t{t}"
        s_var = f"s_{gate_id}_t{t}"
        f_var = f"f_{gate_id}_t{t}"
        
        # Store variables
        if t not in self.time_control_vars:
            self.time_control_vars[t] = []
            self.time_select_vars[t] = []
        
        self.time_control_vars[t].append(c_var)
        self.time_select_vars[t].append(s_var)
        self.gate_fault_vars[gate_id][t] = f_var
        
        # Add fault gadget to DAG
        gate_data = self.dag.nodes[gate_id]
        fault_type = next(iter(self.fault_model.fault_types))  # For now, use first fault type
        
        # Create fault gadget node
        gadget_id = f"{gate_id}_fault_t{t}"
        self.dag.add_node(gadget_id, 
                         type="fault_gadget",
                         control_var=c_var,
                         select_var=s_var,
                         fault_var=f_var,
                         fault_type=fault_type.value,
                         time=t)
        
        # Connect gadget to original gate
        self.dag.add_edge(gate_id, gadget_id)
        
        # Update original gate's output to come from gadget
        for succ in list(self.dag.successors(gate_id)):
            if succ != gadget_id:
                self.dag.add_edge(gadget_id, succ)
                self.dag.remove_edge(gate_id, succ)
    
    def transform(self) -> None:
        """Transform DAG by adding fault gadgets to eligible gates."""
        # Add fault gadgets for each eligible gate at each time step
        for gate_id in self.dag.nodes():
            if self._is_gate_eligible(gate_id):
                for t in range(self.fault_model.max_cycles):
                    self._add_fault_gadget(gate_id, t)
    
    def get_control_vars(self) -> Dict[int, List[str]]:
        """Get control variables indexed by time."""
        return self.time_control_vars
    
    def get_select_vars(self) -> Dict[int, List[str]]:
        """Get select variables indexed by time."""
        return self.time_select_vars
    
    def get_fault_vars(self) -> Dict[str, Dict[int, str]]:
        """Get fault variables indexed by gate and time."""
        return self.gate_fault_vars

# Example usage
if __name__ == "__main__":
    from verilog_parser import parse_verilog_to_dag
    
    # Parse and transform the circuit
    dag = parse_verilog_to_dag("test_circuit.v")
    transformer = GadgetTransformer(dag.graph, dag.fault_model)
    transformer.transform()
    
    # Encode to CNF
    encoder = CNFEncoder(dag)
    encoder.add_constraint_max_faults(3)  # Limit to 3 faults
    cnf = encoder.get_cnf()
    var_map = encoder.get_var_map()
    
    # Solve and analyze
    solve_and_report(cnf, var_map, key_signals=['output', 'fault_control']) 