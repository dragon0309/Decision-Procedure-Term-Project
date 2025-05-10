from pysat.solvers import Solver
from pysat.formula import CNF
import time
from typing import Dict, Optional, Tuple, List
import logging

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

def solve_and_report(cnf: CNF, var_map: Dict[int, str], 
                    key_signals: Optional[List[str]] = None,
                    solver_name: str = "glucose3") -> None:
    """
    Convenience function to solve a CNF formula and print a report.
    
    Args:
        cnf: The CNF formula to solve
        var_map: Mapping from variable numbers to signal names
        key_signals: List of signal names to report (if None, reports all signals)
        solver_name: Name of the SAT solver backend to use
    """
    solver = SATSolver(solver_name)
    solver.solve_and_report(cnf, var_map, key_signals) 