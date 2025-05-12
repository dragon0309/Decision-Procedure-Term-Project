from typing import Dict, List, Set, Optional
from dataclasses import dataclass

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

@dataclass
class CNFClause:
    """Represents a CNF clause with optional comment."""
    literals: List[int]
    comment: Optional[str] = None 