from dataclasses import dataclass
from typing import Set, List, Optional
from enum import Enum

class FaultType(Enum):
    BIT_FLIP = "bit_flip"
    SET_1 = "set_1"
    SET_0 = "set_0"

class GateType(Enum):
    LOGIC = "logic"
    MEMORY = "memory"
    BOTH = "both"

@dataclass
class FaultModel:
    """Fault model parameters for formal verification."""
    max_faults_per_cycle: int  # ne
    max_cycles: int  # nc
    fault_types: Set[FaultType]  # T
    gate_types: GateType  # ℓ
    
    def __post_init__(self):
        """Validate fault model parameters."""
        if self.max_faults_per_cycle < 1:
            raise ValueError("max_faults_per_cycle must be at least 1")
        if self.max_cycles < 1:
            raise ValueError("max_cycles must be at least 1")
        if not self.fault_types:
            raise ValueError("At least one fault type must be specified")
        if not isinstance(self.gate_types, GateType):
            raise ValueError("gate_types must be a GateType enum value")
    
    @classmethod
    def from_dict(cls, params: dict) -> 'FaultModel':
        """Create a FaultModel from a dictionary of parameters."""
        fault_types = {FaultType(ft) for ft in params.get('fault_types', ['bit_flip'])}
        gate_types = GateType(params.get('gate_types', 'both'))
        
        return cls(
            max_faults_per_cycle=params.get('max_faults_per_cycle', 1),
            max_cycles=params.get('max_cycles', 1),
            fault_types=fault_types,
            gate_types=gate_types
        )
    
    def to_dict(self) -> dict:
        """Convert fault model to dictionary."""
        return {
            'max_faults_per_cycle': self.max_faults_per_cycle,
            'max_cycles': self.max_cycles,
            'fault_types': [ft.value for ft in self.fault_types],
            'gate_types': self.gate_types.value
        }
    
    def is_gate_type_allowed(self, gate_type: str) -> bool:
        """Check if a gate type is allowed in this fault model."""
        if self.gate_types == GateType.BOTH:
            return True
        return gate_type == self.gate_types.value 