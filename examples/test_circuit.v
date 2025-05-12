// 

module test_circuit(
    input wire a,
    input wire b,
    output wire out,
    output wire out_protected
);

    // Original circuit
    wire w1, w2;
    and g1(w1, a, b);
    or g2(w2, a, b);
    xor g3(out, w1, w2);

    // Protected circuit (with redundancy)
    wire w1_p, w2_p;
    and g1_p(w1_p, a, b);
    or g2_p(w2_p, a, b);
    xor g3_p(out_protected, w1_p, w2_p);

endmodule