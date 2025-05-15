/**
 * LFSR (Linear Feedback Shift Register) Stream Cipher
 * Generates a pseudo-random keystream for encryption
 */
module lfsr (
    input wire clk,           // Clock signal
    input wire rst,           // Reset signal
    input wire enable,        // Enable signal
    input wire [7:0] seed,    // 8-bit seed value
    input wire load_seed,     // Load new seed
    output wire [7:0] keystream // 8-bit keystream output
);
    // 8-bit LFSR register
    reg [7:0] lfsr_reg;
    
    // LFSR output is the current register value
    assign keystream = lfsr_reg;
    
    // Calculate feedback value using polynomial x^8 + x^6 + x^5 + x^4 + 1
    wire feedback, feedback_1, feedback_2;
    xor xfb1 (feedback_1, lfsr_reg[7], lfsr_reg[5]);
    xor xfb2 (feedback_2, feedback_1, lfsr_reg[4]);
    xor xfb3 (feedback, feedback_2, lfsr_reg[3]);
    
    // LFSR update logic
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            lfsr_reg <= 8'h01;  // Reset to non-zero initial value
        end else if (load_seed) begin
            lfsr_reg <= seed;   // Load new seed
        end else if (enable) begin
            // Shift and insert feedback value
            lfsr_reg <= {lfsr_reg[6:0], feedback};
        end
    end
    
    // Ensure LFSR doesn't enter all-zero state
    always @(lfsr_reg) begin
        if (lfsr_reg == 8'h00) begin
            lfsr_reg <= 8'h01;  // If all-zero detected, force to non-zero value
        end
    end
endmodule 