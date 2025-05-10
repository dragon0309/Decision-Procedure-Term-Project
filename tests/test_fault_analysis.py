import logging
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core import (
    parse_verilog_to_dag,
    GadgetTransformer,
    CNFEncoder,
    solve_and_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_circuit(verilog_file: str, max_faults: int = 3):
    """
    Analyze a Verilog circuit for fault injection vulnerabilities.
    
    Args:
        verilog_file: Path to the Verilog file
        max_faults: Maximum number of faults to consider
    """
    logger.info(f"Analyzing circuit: {verilog_file}")
    
    # Step 1: Parse Verilog to DAG
    logger.info("Parsing Verilog file...")
    dag = parse_verilog_to_dag(verilog_file)
    print(dag.to_json())
    dag.visualize() 
    
    # Step 2: Transform circuit with fault gadgets
    logger.info("Adding fault gadgets...")
    transformer = GadgetTransformer(dag.graph, dag.fault_points)
    transformed_dag = transformer.transform()
    
    # Step 3: Encode to CNF
    logger.info("Encoding to CNF...")
    encoder = CNFEncoder(transformed_dag)
    encoder.add_constraint_max_faults(max_faults)
    cnf = encoder.get_cnf()
    var_map = encoder.get_var_map()
    
    # Step 4: Solve and analyze
    logger.info("Solving SAT instance...")
    solve_and_report(cnf, var_map, 
                    key_signals=['output', 'fault_control'])
    
    return {
        'dag': dag,
        'transformed_dag': transformed_dag,
        'cnf': cnf,
        'var_map': var_map
    }

def main():
    """Main function to demonstrate the fault analysis workflow."""
    # Example usage with test circuit
    result = analyze_circuit("../examples/test_multi_module.v")
    
    # You can access the results for further analysis
    dag = result['dag']
    transformed_dag = result['transformed_dag']
    cnf = result['cnf']
    var_map = result['var_map']
    
    # Print some statistics
    logger.info(f"Number of gates: {len(dag.graph.nodes)}")
    logger.info(f"Number of fault points: {len(dag.fault_points)}")
    logger.info(f"Number of CNF variables: {len(var_map)}")
    logger.info(f"Number of CNF clauses: {len(cnf.clauses)}")

if __name__ == "__main__":
    main() 