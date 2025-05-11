from pysat.solvers import Solver
from pysat.formula import CNF
import time
from typing import Dict, Optional, Tuple, List
import logging
from .cnf_encoder import CNF as CNFEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SATSolver:
    """A wrapper class for PySAT solvers that provides additional functionality for fault injection analysis."""
    
    def __init__(self, solver_name: str = "glucose3"):
        """
        Initialize the SAT solver with the specified backend.
        
        Args:
            solver_name: Name of the SAT solver backend to use (e.g., 'glucose3', 'minisat22')
        """
        self.solver_name = solver_name
        self.solver = None
        self.var_map = None
    
    def solve(self, cnf: CNF, var_map: Dict[int, str]) -> Tuple[bool, Optional[Dict[str, bool]]]:
        """
        Solve a CNF formula and return the result.
        
        Args:
            cnf: The CNF formula to solve
            var_map: Mapping from variable numbers to signal names
            
        Returns:
            Tuple containing:
            - Boolean indicating if the formula is satisfiable
            - Dictionary mapping signal names to their values (if satisfiable)
        """
        self.var_map = var_map
        self.solver = Solver(name=self.solver_name, bootstrap_with=cnf)
        
        start_time = time.time()
        is_sat = self.solver.solve()
        solve_time = time.time() - start_time
        
        logger.info(f"Solving took {solve_time:.2f} seconds")
        
        if is_sat:
            model = self.solver.get_model()
            return True, self._convert_model_to_signal_values(model)
        return False, None
    
    def _convert_model_to_signal_values(self, model: List[int]) -> Dict[str, bool]:
        """Convert the solver's model to a dictionary of signal values."""
        signal_values = {}
        for var in model:
            if abs(var) in self.var_map:
                signal_name = self.var_map[abs(var)]
                signal_values[signal_name] = var > 0
        return signal_values
    
    def solve_and_report(self, cnf: CNF, var_map: Dict[int, str], 
                        key_signals: Optional[List[str]] = None) -> None:
        """
        Solve a CNF formula and print a report of key signal values.
        
        Args:
            cnf: The CNF formula to solve
            var_map: Mapping from variable numbers to signal names
            key_signals: List of signal names to report (if None, reports all signals)
        """
        is_sat, model = self.solve(cnf, var_map)
        
        if is_sat:
            logger.info("Formula is satisfiable")
            if key_signals:
                logger.info("Key signal values:")
                for signal in key_signals:
                    if signal in model:
                        logger.info(f"  {signal}: {model[signal]}")
            else:
                logger.info("All signal values:")
                for signal, value in model.items():
                    logger.info(f"  {signal}: {value}")
        else:
            logger.info("Formula is unsatisfiable")
    
    def __del__(self):
        """Clean up the solver when the object is destroyed."""
        if self.solver:
            self.solver.delete()

def solve_cnf(cnf: CNF, var_map: Dict[int, str], 
             solver_name: str = "glucose3") -> Tuple[bool, Optional[Dict[str, bool]]]:
    """
    Convenience function to solve a CNF formula.
    
    Args:
        cnf: The CNF formula to solve
        var_map: Mapping from variable numbers to signal names
        solver_name: Name of the SAT solver backend to use
        
    Returns:
        Tuple containing:
        - Boolean indicating if the formula is satisfiable
        - Dictionary mapping signal names to their values (if satisfiable)
    """
    solver = SATSolver(solver_name)
    return solver.solve(cnf, var_map)

def solve_and_report(cnf: CNF, var_map: Dict[str, int], keys_to_monitor: List[str] = []) -> Optional[Dict[str, bool]]:
    """
    Solve the CNF formula and report results.
    
    Args:
        cnf: The CNF formula to solve
        var_map: Mapping from signal names to variable IDs
        keys_to_monitor: List of signal names to monitor in the output
        
    Returns:
        Dictionary of signal assignments if SAT, None if UNSAT
    """
    # Initialize solver
    solver = Solver()
    
    # Add clauses
    for clause in cnf.clauses:
        if hasattr(clause, 'literals'):
            # Handle CNFClause objects
            solver.add_clause(clause.literals)
        else:
            # Handle direct clause lists
            solver.add_clause(clause)
    
    # Solve with timing
    start_time = time.time()
    is_sat = solver.solve()
    solve_time = time.time() - start_time
    
    print(f"\nSAT solving time: {solve_time:.3f} seconds")
    
    if is_sat:
        print("\nSuccessful attack found!")
        model = solver.get_model()
        assignments = extract_assignment(model, var_map)
        
        # Print monitored values
        if keys_to_monitor:
            print("\nMonitored signal values:")
            for key in keys_to_monitor:
                if key in assignments:
                    print(f"{key}: {assignments[key]}")
        
        # Analyze fault injection
        fault_analysis = analyze_fault_injection(assignments, var_map)
        if fault_analysis:
            print("\nFault injection analysis:")
            for gate, fault_type in fault_analysis.items():
                print(f"Gate {gate}: {fault_type}")
        
        return assignments
    else:
        print("\nNo successful attack found")
        return None

def extract_assignment(model: List[int], var_map: Dict[str, int]) -> Dict[str, bool]:
    """
    Convert a SAT model (list of literals) to a dictionary of signal assignments.
    
    Args:
        model: List of literals from SAT solver
        var_map: Mapping from signal names to variable IDs
        
    Returns:
        Dictionary mapping signal names to their boolean values
    """
    # Create reverse mapping from var_id to signal name
    reverse_map = {v: k for k, v in var_map.items()}
    
    # Extract assignments
    assignments = {}
    for lit in model:
        var_id = abs(lit)
        if var_id in reverse_map:
            signal_name = reverse_map[var_id]
            assignments[signal_name] = lit > 0
            
    return assignments

def analyze_fault_injection(assignments: Dict[str, bool], var_map: Dict[str, int]) -> Dict[str, str]:
    """
    Analyze which gates were faulted and how.
    
    Args:
        assignments: Dictionary of signal assignments
        var_map: Mapping from signal names to variable IDs
        
    Returns:
        Dictionary mapping gate names to fault types
    """
    fault_analysis = {}
    
    # Look for control variables (they start with 'c')
    for signal, value in assignments.items():
        if signal.startswith('c'):
            # Extract gate name from control variable
            gate_name = signal[1:]  # Remove 'c' prefix
            
            # Determine fault type based on control variable value
            if value:
                fault_analysis[gate_name] = "Fault injected"
            else:
                fault_analysis[gate_name] = "No fault"
    
    return fault_analysis 