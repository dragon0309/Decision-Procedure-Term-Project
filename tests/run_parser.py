#!/usr/bin/env python3
"""
Test script: Run verilog_parser to parse test circuits and output results
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.verilog_parser import parse_verilog_to_dag
from src.core.fault_model import FaultModel, FaultType, GateType

def parse_circuit(circuit_path, visualize=True, use_synthesis=True, reduce_fault_points=True):
    """
    Parse the specified circuit file, generate JSON representation and visualization graph
    
    Args:
        circuit_path: Path to circuit file
        visualize: Whether to generate visualization graph
        use_synthesis: Whether to attempt Yosys synthesis
        reduce_fault_points: Whether to reduce fault points
    """
    print(f"\nParsing circuit: {circuit_path}")
    
    try:
        # Create fault model, allowing both logic and memory gate faults
        fault_model = FaultModel(
            max_faults_per_cycle=2,
            max_cycles=1,
            fault_types={FaultType.BIT_FLIP, FaultType.SET_1, FaultType.SET_0},
            gate_types=GateType.BOTH
        )
        
        # Parse circuit
        dag = parse_verilog_to_dag(
            circuit_path, 
            fault_model=fault_model,
            use_synthesis=use_synthesis,
            reduce_fault_points=reduce_fault_points
        )
        
        # Output JSON representation
        output_dir = os.path.dirname(circuit_path)
        base_name = os.path.splitext(os.path.basename(circuit_path))[0]
        
        # Write JSON file
        json_output = os.path.join(output_dir, f"{base_name}_dag.json")
        with open(json_output, 'w', encoding='utf-8') as f:
            f.write(dag.to_json())
        
        # Extract statistics from JSON
        json_data = json.loads(dag.to_json())
        
        print(f"Number of nodes: {len(json_data['nodes'])}")
        print(f"Number of edges: {len(json_data['edges'])}")
        print(f"Number of fault points: {len(json_data['fault_points'])}")
        print(f"Number of critical gates: {len(json_data['critical_gates'])}")
        print(f"Number of inputs: {len(json_data['inputs'])}")
        print(f"Number of outputs: {len(json_data['outputs'])}")
        
        # Generate visualization
        if visualize:
            vis_output = os.path.join(output_dir, f"{base_name}_circuit")
            print(f"Generating visualization: {vis_output}.png")
            dag.visualize(vis_output)
        
        print(f"Parse complete, results saved to {json_output}")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Main function: Parse all test circuits"""
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Parse test circuits and output results')
    parser.add_argument('--no-vis', action='store_true', help='Do not generate visualization')
    parser.add_argument('--no-synthesis', action='store_true', help='Do not attempt Yosys synthesis')
    parser.add_argument('--keep-all-faults', action='store_true', help='Do not reduce fault points')
    parser.add_argument('--circuit', help='Only parse specified circuit')
    args = parser.parse_args()
    
    # Test circuit list
    test_circuits = [
        "circuits/xor_cipher.v",
        "circuits/shift_cipher.v",
        "circuits/sbox.v",
        "circuits/mixcolumn.v",
        "circuits/lfsr.v"
    ]
    
    # Tests directory path
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Parse specified or all circuits
    if args.circuit:
        circuit_path = os.path.join(tests_dir, args.circuit)
        if not os.path.exists(circuit_path):
            print(f"Circuit file not found: {circuit_path}")
            return 1
        
        success = parse_circuit(
            circuit_path, 
            visualize=not args.no_vis,
            use_synthesis=not args.no_synthesis,
            reduce_fault_points=not args.keep_all_faults
        )
        
        return 0 if success else 1
    else:
        all_success = True
        for circuit in test_circuits:
            circuit_path = os.path.join(tests_dir, circuit)
            if not os.path.exists(circuit_path):
                print(f"Circuit file not found: {circuit_path}")
                all_success = False
                continue
                
            success = parse_circuit(
                circuit_path, 
                visualize=not args.no_vis,
                use_synthesis=not args.no_synthesis, 
                reduce_fault_points=not args.keep_all_faults
            )
            
            if not success:
                all_success = False
        
        return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main()) 