# Verilog Fault Analysis Tool Test Suite

This test suite contains multiple test circuits and scripts to verify the functionality of the Verilog Fault Analysis Tool.

## Test Circuits

The tests directory contains 5 simple cryptographic circuit implementations:

1. **XOR Cipher (xor_cipher.v)**:
   - 8-bit data XORed with 8-bit key
   - Includes synchronous elements (output registers)
   - Simple but practical encryption primitive

2. **Shift Cipher (shift_cipher.v)**:
   - 8-bit data left-shifted based on 3-bit key value
   - Contains both synchronous elements and combinational logic
   - Similar to a simple Caesar cipher

3. **S-Box Substitution (sbox.v)**:
   - 4-bit input to 4-bit output lookup table
   - Similar to the non-linear transformation used in AES and other modern ciphers
   - Pure combinational logic implementation

4. **Simplified AES MixColumn (mixcolumn.v)**:
   - Implements a simplified version of matrix multiplication in GF(2^8)
   - Similar to the MixColumns operation in AES
   - Uses XOR gates to implement Galois field operations

5. **LFSR Stream Cipher (lfsr.v)**:
   - 8-bit Linear Feedback Shift Register
   - Generates a pseudo-random keystream
   - Includes synchronous elements and feedback paths

## Running Tests

### Run All Tests

```bash
python tests/run_parser.py
```

### Run a Specific Circuit Test

```bash
python tests/run_parser.py --circuit circuits/xor_cipher.v
```

### Available Options

- `--no-vis`: Do not generate visualization graphs
- `--no-synthesis`: Do not attempt to use Yosys for synthesis (parse Verilog directly)
- `--keep-all-faults`: Do not reduce fault points (keep all possible fault points)
- `--circuit <path>`: Only parse the specified circuit

## Test Output Explanation

After running tests, each circuit will produce the following outputs:

1. **JSON File** (`<circuit_name>_dag.json`):
   - Node list: All gates and signals in the circuit
   - Edge list: All connections between gates and signals
   - Fault point list: Points where faults can be injected
   - Critical gates list: Gates that affect outputs
   - Module hierarchy
   - Input and output lists

2. **Visualization File** (`<circuit_name>_circuit.png`):
   - Visual representation of the circuit graph
   - Red nodes: Potential fault injection points
   - Blue nodes: Critical gates (affecting outputs)
   - Black nodes: Other circuit elements

3. **Console Output**:
   - Number of nodes
   - Number of edges
   - Number of fault points
   - Number of critical gates
   - Number of inputs
   - Number of outputs

## Circuit Fault Analysis Result Interpretation

- **Fault Points**: Nodes in each circuit where faults can be injected. Based on the fault model, only nodes of logic and memory categories are considered potential fault points
- **Critical Gates**: Nodes that affect outputs. Only faults in critical gates will cause output changes
- **Reduced Fault Set**: Intersection of fault points and critical gates, which are the nodes that truly need protection

## Fault Model Explanation

Fault model settings used in testing:
- Maximum of 2 faults per cycle
- Maximum of 1 cycle
- Supported fault types: bit flip, set to 1, set to 0
- Allowed gate types: both logic gates and memory elements 