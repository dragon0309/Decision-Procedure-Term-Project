/**
 * S-Box Substitution Circuit
 * Implements a 4-bit to 4-bit lookup table, similar to AES S-Box
 */
module sbox (
    input wire clk,           // Clock signal
    input wire rst,           // Reset signal
    input wire [3:0] data,    // 4-bit input data
    output reg [3:0] out      // 4-bit transformed output
);
    // Combinational logic - implements simplified S-Box lookup
    wire [3:0] sbox_result;
    
    // Lookup corresponding S-Box value based on input
    // Implements a typical non-linear transformation using XOR, AND, OR, NOT operations
    wire [3:0] temp1, temp2;
    
    // First layer of non-linear transformation - using AND and XOR
    and a1 (temp1[0], data[0], data[1]);
    xor x1 (temp1[1], data[1], data[2]);
    xor x2 (temp1[2], data[2], data[3]);
    and a2 (temp1[3], data[3], data[0]);
    
    // Second layer of non-linear transformation - using XOR and NOT
    wire not_data0, not_data1, not_data2, not_data3;
    not n1 (not_data0, data[0]);
    not n2 (not_data1, data[1]);
    not n3 (not_data2, data[2]);
    not n4 (not_data3, data[3]);
    
    xor x3 (temp2[0], temp1[0], not_data2);
    xor x4 (temp2[1], temp1[1], not_data3);
    xor x5 (temp2[2], temp1[2], not_data0);
    xor x6 (temp2[3], temp1[3], not_data1);
    
    // Final transformation - using OR and XOR
    wire [3:0] temp3;
    or o1 (temp3[0], temp1[0], temp2[3]);
    or o2 (temp3[1], temp1[1], temp2[0]);
    or o3 (temp3[2], temp1[2], temp2[1]);
    or o4 (temp3[3], temp1[3], temp2[2]);
    
    xor x7 (sbox_result[0], temp3[0], data[2]);
    xor x8 (sbox_result[1], temp3[1], data[3]);
    xor x9 (sbox_result[2], temp3[2], data[0]);
    xor x10 (sbox_result[3], temp3[3], data[1]);
    
    // Sequential logic to store S-Box result
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            out <= 4'b0;  // Reset output to 0
        end else begin
            out <= sbox_result;  // Store S-Box result
        end
    end
endmodule 