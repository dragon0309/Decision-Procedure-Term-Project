# Verilog Fault Analysis Tool

A Python-based tool for analyzing fault injection vulnerabilities in Verilog circuits.

## Project Structure

```
.
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── verilog_parser.py
│   │   ├── gadget_transformer.py
│   │   ├── cnf_encoder.py
│   │   └── sat_solver.py
│   └── utils/
├── tests/
│   └── test_fault_analysis.py
├── examples/
│   ├── test_circuit.v
│   └── test_multi_module.v
├── docs/
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```python
from src.core import parse_verilog_to_dag, GadgetTransformer, CNFEncoder, solve_and_report

# Parse and analyze a circuit
dag = parse_verilog_to_dag("examples/test_circuit.v")
transformer = GadgetTransformer(dag.graph, dag.fault_points)
transformed_dag = transformer.transform()

# Encode to CNF and solve
encoder = CNFEncoder(transformed_dag)
encoder.add_constraint_max_faults(3)
cnf = encoder.get_cnf()
var_map = encoder.get_var_map()

# Solve and analyze
solve_and_report(cnf, var_map, key_signals=['output', 'fault_control'])
```

## Features

- Verilog circuit parsing
- Fault point identification
- Fault gadget transformation
- CNF encoding
- SAT solving
- Result analysis

## Requirements

- Python 3.6+
- pyverilog
- networkx
- graphviz
- python-sat

## License

MIT License 