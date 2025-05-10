module and_gate(
    input wire a,
    input wire b,
    output wire y
);
    and (y, a, b);
endmodule

module or_gate(
    input wire a,
    input wire b,
    output wire y
);
    or (y, a, b);
endmodule

module complex_gate(
    input wire [1:0] in,
    output wire out
);
    wire w1, w2;
    
    and_gate and1(.a(in[0]), .b(in[1]), .y(w1));
    or_gate or1(.a(in[0]), .b(in[1]), .y(w2));
    xor (out, w1, w2);
endmodule

module top(
    input wire [1:0] in,
    output wire out,
    output wire [1:0] state
);
    wire w1;
    
    complex_gate cg(.in(in), .out(w1));
    dff d1(.D(w1), .Q(state[0]), .CLK(in[0]));
    dff d2(.D(state[0]), .Q(state[1]), .CLK(in[1]));
    assign out = state[1];
endmodule 