/**
 * XOR Cipher Circuit
 * 8-bit data XORed with 8-bit key, result stored in register
 */
module xor_cipher (
    input wire clk,
    input wire rst,
    input wire [7:0] data,    // 8-bit input data
    input wire [7:0] key,     // 8-bit key
    output reg [7:0] out      // 8-bit encrypted output
);
    // Internal XOR result
    wire [7:0] xor_result;
    
    // XOR each bit
    xor x0 (xor_result[0], data[0], key[0]);
    xor x1 (xor_result[1], data[1], key[1]);
    xor x2 (xor_result[2], data[2], key[2]);
    xor x3 (xor_result[3], data[3], key[3]);
    xor x4 (xor_result[4], data[4], key[4]);
    xor x5 (xor_result[5], data[5], key[5]);
    xor x6 (xor_result[6], data[6], key[6]);
    xor x7 (xor_result[7], data[7], key[7]);
    
    // Sequential logic to store result
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            out <= 8'b0;  // Reset output to 0
        end else begin
            out <= xor_result;  // Store XOR result
        end
    end
endmodule 