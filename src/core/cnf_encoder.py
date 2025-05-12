from typing import Dict, List, Set, Any, Optional, Tuple
import networkx as nx
from pysat.formula import CNF
from .models import FaultModel, CNFClause

class CNFEncoder:
    """Encodes a circuit DAG into CNF format."""
    
    def __init__(self, dag: nx.DiGraph, fault_model: Optional[FaultModel] = None):
        """
        Initialize the encoder.
        
        Args:
            dag: Circuit DAG
            fault_model: Fault injection model parameters
        """
        self.dag = dag
        self.fault_model = fault_model or FaultModel(
            max_faults_per_cycle=3,
            max_cycles=1,
            allowed_fault_types={"bit_flip", "set_1", "set_0"},
            allowed_gate_types={"logic", "memory"}
        )
        self.var_map: Dict[str, int] = {}  # wire/gate -> SAT variable
        self.clauses: List[CNFClause] = []
        self.var_counter = 1  # Start from 1 (0 is reserved)
        self._cnf = None  # Cache for CNF formula
    
    def get_cnf(self) -> CNF:
        """Get the CNF formula."""
        if self._cnf is None:
            self._cnf, _ = self.encode()
        return self._cnf
    
    def get_var_map(self) -> Dict[str, int]:
        """Get the variable mapping."""
        if not self.var_map:
            self.encode()  # This will populate var_map
        return self.var_map
    
    def encode(self) -> Tuple[CNF, Dict[str, int]]:
        """Encode the circuit into CNF format."""
        # First pass: assign variables to all wires and gates
        self._assign_variables()
        
        # Second pass: generate clauses for each gate
        for node in self.dag.nodes():
            self._encode_gate(node)
        
        # Create CNF formula
        cnf = CNF()
        for clause in self.clauses:
            cnf.append(clause.literals)
        
        return cnf, self.var_map
    
    def _assign_variables(self) -> None:
        """Assign SAT variables to all wires and gates."""
        # First pass: assign variables to all nodes and their inputs/outputs
        for node in self.dag.nodes():
            node_data = self.dag.nodes[node]
            
            # Assign variable to the gate itself
            if node not in self.var_map:
                self.var_map[node] = self.var_counter
                self.var_counter += 1
            
            # Assign variables to all inputs
            for inp in node_data["inputs"]:
                if inp not in self.var_map:
                    self.var_map[inp] = self.var_counter
                    self.var_counter += 1
            
            # Assign variable to output
            if "output" in node_data and node_data["output"] not in self.var_map:
                self.var_map[node_data["output"]] = self.var_counter
                self.var_counter += 1
        
        # Second pass: assign variables to all edges (wires)
        for edge in self.dag.edges():
            wire = f"{edge[0]}_{edge[1]}"
            if wire not in self.var_map:
                self.var_map[wire] = self.var_counter
                self.var_counter += 1
    
    def _encode_gate(self, gate_id: str) -> None:
        """Generate CNF clauses for a specific gate."""
        gate_data = self.dag.nodes[gate_id]
        gate_type = gate_data["type"]
        
        # Ensure all inputs and output are in the variable map
        for inp in gate_data["inputs"]:
            if inp not in self.var_map:
                self.var_map[inp] = self.var_counter
                self.var_counter += 1
        
        if gate_id not in self.var_map:
            self.var_map[gate_id] = self.var_counter
            self.var_counter += 1
            
        if "output" in gate_data and gate_data["output"] not in self.var_map:
            self.var_map[gate_data["output"]] = self.var_counter
            self.var_counter += 1
        
        # Get input and output variables
        inputs = [self.var_map[inp] for inp in gate_data["inputs"]]
        output = self.var_map[gate_id]
        
        # Generate clauses based on gate type
        if gate_type == "and":
            self._encode_and_gate(inputs, output, gate_id)
        elif gate_type == "or":
            self._encode_or_gate(inputs, output, gate_id)
        elif gate_type == "xor":
            self._encode_xor_gate(inputs, output, gate_id)
        elif gate_type == "not":
            self._encode_not_gate(inputs[0], output, gate_id)
        elif gate_type == "dff":
            self._encode_dff_gate(inputs, output, gate_id)
        
        # Add fault gadget clauses if present
        if "fault_control" in gate_data:
            self._encode_fault_gadget(gate_id)
    
    def _encode_and_gate(self, inputs: List[int], output: int, gate_id: str) -> None:
        """Encode AND gate: (A & B) -> Y"""
        # Y -> (A & B)
        self.clauses.append(CNFClause(
            literals=[-output] + inputs,
            comment=f"AND gate {gate_id}: Y -> (A & B)"
        ))
        # (A & B) -> Y
        for inp in inputs:
            self.clauses.append(CNFClause(
                literals=[-inp, output],
                comment=f"AND gate {gate_id}: A -> Y"
            ))
    
    def _encode_or_gate(self, inputs: List[int], output: int, gate_id: str) -> None:
        """Encode OR gate: (A | B) -> Y"""
        # Y -> (A | B)
        for inp in inputs:
            self.clauses.append(CNFClause(
                literals=[-output, inp],
                comment=f"OR gate {gate_id}: Y -> A"
            ))
        # (A | B) -> Y
        self.clauses.append(CNFClause(
            literals=[output] + [-inp for inp in inputs],
            comment=f"OR gate {gate_id}: (A | B) -> Y"
        ))
    
    def _encode_xor_gate(self, inputs: List[int], output: int, gate_id: str) -> None:
        """Encode XOR gate: (A ^ B) -> Y"""
        # Y -> (A ^ B)
        self.clauses.append(CNFClause(
            literals=[-output, inputs[0], inputs[1]],
            comment=f"XOR gate {gate_id}: Y -> (A ^ B) 1"
        ))
        self.clauses.append(CNFClause(
            literals=[-output, -inputs[0], -inputs[1]],
            comment=f"XOR gate {gate_id}: Y -> (A ^ B) 2"
        ))
        # (A ^ B) -> Y
        self.clauses.append(CNFClause(
            literals=[output, -inputs[0], -inputs[1]],
            comment=f"XOR gate {gate_id}: (A ^ B) -> Y 1"
        ))
        self.clauses.append(CNFClause(
            literals=[output, inputs[0], inputs[1]],
            comment=f"XOR gate {gate_id}: (A ^ B) -> Y 2"
        ))
    
    def _encode_not_gate(self, input_var: int, output: int, gate_id: str) -> None:
        """Encode NOT gate: ~A -> Y"""
        # Y <-> ~A
        self.clauses.append(CNFClause(
            literals=[-output, -input_var],
            comment=f"NOT gate {gate_id}: Y -> ~A"
        ))
        self.clauses.append(CNFClause(
            literals=[output, input_var],
            comment=f"NOT gate {gate_id}: ~A -> Y"
        ))
    
    def _encode_dff_gate(self, inputs: List[int], output: int, gate_id: str) -> None:
        """Encode DFF: D -> Q (next cycle)"""
        # For DFF, we need to handle the clock and state
        # This is a simplified version - in practice, you'd need to handle
        # the clock domain and state transitions properly
        self.clauses.append(CNFClause(
            literals=[-inputs[0], output],  # D -> Q
            comment=f"DFF {gate_id}: D -> Q"
        ))
        self.clauses.append(CNFClause(
            literals=[inputs[0], -output],  # ~D -> ~Q
            comment=f"DFF {gate_id}: ~D -> ~Q"
        ))
    
    def _encode_fault_gadget(self, gate_id: str) -> None:
        """Encode fault injection gadget."""
        gate_data = self.dag.nodes[gate_id]
        
        # Ensure fault control and select variables are in the variable map
        control_var_name = gate_data["fault_control"]
        select_var_name = gate_data["fault_select"]
        
        if control_var_name not in self.var_map:
            self.var_map[control_var_name] = self.var_counter
            self.var_counter += 1
        if select_var_name not in self.var_map:
            self.var_map[select_var_name] = self.var_counter
            self.var_counter += 1
            
        control_var = self.var_map[control_var_name]
        select_var = self.var_map[select_var_name]
        output = self.var_map[gate_id]
        
        # Get original output if available, otherwise use gate output
        original_output = self.var_map.get(gate_data.get("original_output", gate_id))
        
        # Fault injection logic:
        # output = (control_var ? faulty_output : original_output)
        self.clauses.append(CNFClause(
            literals=[-control_var, -output, original_output],
            comment=f"Fault gadget {gate_id}: control=0 -> original"
        ))
        self.clauses.append(CNFClause(
            literals=[-control_var, output, -original_output],
            comment=f"Fault gadget {gate_id}: control=0 -> original"
        ))
        
        # Add clauses for different fault types
        for fault_type in gate_data["fault_types"]:
            if fault_type == "bit_flip":
                self.clauses.append(CNFClause(
                    literals=[control_var, -select_var, -output, -original_output],
                    comment=f"Fault gadget {gate_id}: bit flip"
                ))
                self.clauses.append(CNFClause(
                    literals=[control_var, -select_var, output, original_output],
                    comment=f"Fault gadget {gate_id}: bit flip"
                ))
            elif fault_type == "set_1":
                self.clauses.append(CNFClause(
                    literals=[control_var, select_var, output],
                    comment=f"Fault gadget {gate_id}: set to 1"
                ))
            elif fault_type == "set_0":
                self.clauses.append(CNFClause(
                    literals=[control_var, select_var, -output],
                    comment=f"Fault gadget {gate_id}: set to 0"
                ))
    
    def to_dimacs(self, filename: str) -> None:
        """Write CNF formula to DIMACS format."""
        cnf, _ = self.encode()
        cnf.to_file(filename)
    
    def add_constraint_max_faults(self, max_faults: int) -> None:
        """Add constraint: sum of control variables <= max_faults per cycle."""
        # Get control variables from nodes that have fault gadgets
        for t in range(self.fault_model.max_cycles):
            cycle_control_vars = []
            for node in self.dag.nodes():
                node_data = self.dag.nodes[node]
                if "fault_control" in node_data:
                    control_var = f"{node_data['fault_control']}_t{t}"
                    if control_var in self.var_map:
                        cycle_control_vars.append(self.var_map[control_var])
            
            # Add at-most-k constraint for this cycle
            for i in range(len(cycle_control_vars)):
                for j in range(i + 1, len(cycle_control_vars)):
                    if i + j >= max_faults:
                        self.clauses.append(CNFClause(
                            literals=[-cycle_control_vars[i], -cycle_control_vars[j]],
                            comment=f"At-most-{max_faults} constraint for cycle {t}"
                        ))

    def add_nc_time_constraint(self, time_control_vars: Dict[int, List[str]], max_active_cycles: int) -> None:
        """
        Add constraint to limit the number of active fault injection cycles.
        
        Args:
            time_control_vars: Dictionary mapping time indices to lists of control variables
            max_active_cycles: Maximum number of cycles where faults can be active
        """
        # Create variables for each time slice indicating if any fault is active
        time_active_vars = {}
        for t in time_control_vars:
            time_active_var = f"active_t{t}"
            self.var_map[time_active_var] = self.var_counter
            time_active_vars[t] = self.var_counter
            self.var_counter += 1
            
            # Ensure all control variables are in the variable map
            control_vars = time_control_vars[t]
            control_var_ids = []
            for cv in control_vars:
                if cv not in self.var_map:
                    self.var_map[cv] = self.var_counter
                    self.var_counter += 1
                control_var_ids.append(self.var_map[cv])
            
            # Encode: time_active_var ↔ (control_var1 ∨ control_var2 ∨ ...)
            # Add clauses for OR condition
            for cv_id in control_var_ids:
                self.clauses.append(CNFClause(
                    literals=[-cv_id, time_active_vars[t]],
                    comment=f"Time {t} active if control var {cv_id} is active"
                ))
            
            self.clauses.append(CNFClause(
                literals=[-time_active_vars[t]] + control_var_ids,
                comment=f"Time {t} active only if some control var is active"
            ))
        
        # Add at-most-k constraint on active cycles
        active_cycle_vars = list(time_active_vars.values())
        for i in range(len(active_cycle_vars)):
            for j in range(i + 1, len(active_cycle_vars)):
                if i + j >= max_active_cycles:
                    self.clauses.append(CNFClause(
                        literals=[-active_cycle_vars[i], -active_cycle_vars[j]],
                        comment=f"At-most-{max_active_cycles} active cycles"
                    )) 