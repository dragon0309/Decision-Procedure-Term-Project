/**
 * Shift Cipher Circuit
 * 8-bit data shifted based on 3-bit key value
 */
module shift_cipher (
    input wire clk,           // Clock signal
    input wire rst,           // Reset signal
    input wire [7:0] data,    // 8-bit input data
    input wire [2:0] key,     // 3-bit key determines shift amount
    output reg [7:0] out      // 8-bit encrypted output
);
    // Internal variables
    wire [7:0] shift_result;  // Shift result
    
    // Shift section - using multiplexers to implement variable shifting
    wire [7:0] shift1, shift2, shift4;
    
    // 1-bit shift
    assign shift1 = key[0] ? {data[6:0], data[7]} : data;
    
    // 2-bit shift
    assign shift2 = key[1] ? {shift1[5:0], shift1[7:6]} : shift1;
    
    // 4-bit shift
    assign shift4 = key[2] ? {shift2[3:0], shift2[7:4]} : shift2;
    
    // Final result
    assign shift_result = shift4;
    
    // Sequential logic to store shift result
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            out <= 8'b0;  // Reset output to 0
        end else begin
            out <= shift_result;  // Store shift result
        end
    end
endmodule 