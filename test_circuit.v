module test_circuit(
    input clk,
    input rst,
    input [7:0] data_in,
    output [7:0] data_out
);

    // Internal registers
    reg [7:0] reg1, reg2;
    
    // Logic gates
    wire [7:0] xor_out = reg1 ^ reg2;
    wire [7:0] and_out = reg1 & reg2;
    wire [7:0] or_out = reg1 | reg2;
    
    // DFF for state
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            reg1 <= 8'h00;
            reg2 <= 8'h00;
        end else begin
            reg1 <= data_in;
            reg2 <= xor_out;
        end
    end
    
    // Output assignment
    assign data_out = and_out;

endmodule 