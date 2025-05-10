// 

module test_circuit(
    input wire a,
    input wire b,
    input wire c,
    output wire out
);

    wire w1, w2;
    
    // AND gate
    and g1(w1, a, b);
    
    // OR gate
    or g2(w2, w1, c);
    
    // NOT gate
    not g3(out, w2);

endmodule