/**
 * Simplified AES MixColumn Circuit
 * Implements finite field operations in GF(2^8), similar to AES MixColumns
 */
module mixcolumn (
    input wire clk,           // Clock signal
    input wire rst,           // Reset signal
    input wire [7:0] data,    // 8-bit input data
    output reg [7:0] out      // 8-bit mixed output
);
    // Implement simplified GF(2^8) multiplication
    wire [7:0] mix_result;
    
    // Multiply by 2 in GF(2^8), equivalent to left shift with conditional XOR with 0x1B
    wire [7:0] mul2;  // Result of multiplication by 2
    wire overflow;    // Determines if XOR with 0x1B is needed
    
    assign overflow = data[7];
    assign mul2 = {data[6:0], 1'b0} ^ (overflow ? 8'h1B : 8'h00);
    
    // Multiply by 3 in GF(2^8), equivalent to multiply by 2 then XOR with original value
    wire [7:0] mul3;
    assign mul3 = mul2 ^ data;
    
    // Simplified MixColumn operation: apply specific linear transformation to 8-bit data
    // For 8-bit data [a, b, c, d, e, f, g, h], we divide into two groups [a, b, c, d] and [e, f, g, h]
    // Apply AES-like operation to each group
    
    wire [3:0] high, low;
    assign high = data[7:4];
    assign low = data[3:0];
    
    // Apply MixColumn-like operation to both groups
    wire [3:0] high_transformed, low_transformed;
    
    // High 4 bits transformation
    xor x1 (high_transformed[0], high[0], high[1]);
    xor x2 (high_transformed[1], high[1], high[2]);
    xor x3 (high_transformed[2], high[2], high[3]);
    xor x4 (high_transformed[3], high[3], high[0]);
    
    // Low 4 bits transformation
    xor x5 (low_transformed[0], low[0], low[3]);
    xor x6 (low_transformed[1], low[1], low[0]);
    xor x7 (low_transformed[2], low[2], low[1]);
    xor x8 (low_transformed[3], low[3], low[2]);
    
    // Cross operations between groups, simulating the diffusion property of MixColumns
    wire [7:0] cross;
    xor xc1 (cross[0], high[0], low[0]);
    xor xc2 (cross[1], high[1], low[1]);
    xor xc3 (cross[2], high[2], low[2]);
    xor xc4 (cross[3], high[3], low[3]);
    xor xc5 (cross[4], high[0], low[1]);
    xor xc6 (cross[5], high[1], low[2]);
    xor xc7 (cross[6], high[2], low[3]);
    xor xc8 (cross[7], high[3], low[0]);
    
    // Final result is a combination of all transformations
    xor xr1 (mix_result[0], high_transformed[0], cross[0]);
    xor xr2 (mix_result[1], high_transformed[1], cross[1]);
    xor xr3 (mix_result[2], high_transformed[2], cross[2]);
    xor xr4 (mix_result[3], high_transformed[3], cross[3]);
    xor xr5 (mix_result[4], low_transformed[0], cross[4]);
    xor xr6 (mix_result[5], low_transformed[1], cross[5]);
    xor xr7 (mix_result[6], low_transformed[2], cross[6]);
    xor xr8 (mix_result[7], low_transformed[3], cross[7]);
    
    // Sequential logic to store the mixed result
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            out <= 8'b0;  // Reset output to 0
        end else begin
            out <= mix_result;  // Store mixed result
        end
    end
endmodule 