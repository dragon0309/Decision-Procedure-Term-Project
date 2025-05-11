from typing import List, Tuple
from dataclasses import dataclass
from .cnf_encoder import CNFEncoder, CNFClause

def add_output_mismatch_clauses(cnf_encoder: CNFEncoder, output_pairs: List[Tuple[str, str]], oflag_var: str) -> None:
    """
    Add CNF clauses that model output mismatch detection.
    
    Args:
        cnf_encoder: The CNF encoder instance
        output_pairs: List of (original_out, protected_out) pairs to compare
        oflag_var: Name of the output flag variable
    """
    # Ensure oflag variable is in the variable map
    if oflag_var not in cnf_encoder.var_map:
        cnf_encoder.var_map[oflag_var] = cnf_encoder.var_counter
        cnf_encoder.var_counter += 1
    
    oflag_id = cnf_encoder.var_map[oflag_var]
    
    # For each output pair, create mismatch detection clauses
    mismatch_clauses = []
    for orig_out, prot_out in output_pairs:
        # Ensure both variables are in the map
        if orig_out not in cnf_encoder.var_map:
            cnf_encoder.var_map[orig_out] = cnf_encoder.var_counter
            cnf_encoder.var_counter += 1
        if prot_out not in cnf_encoder.var_map:
            cnf_encoder.var_map[prot_out] = cnf_encoder.var_counter
            cnf_encoder.var_counter += 1
            
        orig_id = cnf_encoder.var_map[orig_out]
        prot_id = cnf_encoder.var_map[prot_out]
        
        # XOR encoding for mismatch: (¬o1 ∨ o2) ∧ (o1 ∨ ¬o2)
        cnf_encoder.clauses.append(CNFClause(
            literals=[-orig_id, prot_id],
            comment=f"Output mismatch {orig_out}≠{prot_out} (1)"
        ))
        cnf_encoder.clauses.append(CNFClause(
            literals=[orig_id, -prot_id],
            comment=f"Output mismatch {orig_out}≠{prot_out} (2)"
        ))
        
        # Add to mismatch clauses for aggregation
        mismatch_clauses.extend([-orig_id, prot_id, orig_id, -prot_id])
    
    # Add clause for undetected attack: (output_mismatch_detected) ∧ (oflag == 0)
    # This means at least one output pair must mismatch AND oflag must be 0
    cnf_encoder.clauses.append(CNFClause(
        literals=mismatch_clauses + [-oflag_id],
        comment="Undetected attack condition"
    )) 