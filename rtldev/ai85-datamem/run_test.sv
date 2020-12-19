// run_test.sv
`define ARM_PROG_SOURCE test.c
// Timeout: 180 ms
defparam REPEAT_TIMEOUT = 18.0;

`define CNN_ENA  tb.xchip.xuut1.x16proc[0].xproc.xuut.cnnena
`define CNN_CLK  tb.xchip.xuut1.x16proc[0].xproc.xuut.clk

real  start_time;
real  end_time;
real  clk1_time;
real  clk2_time;
logic start_ena;
logic clkena1;
logic clkena2;

initial begin
   start_time = 0;
   end_time   = 0;
   clk1_time  = 0;
   clk2_time  = 0;
   start_ena  = 0;
   clkena1    = 0;
   clkena2    = 0;
end

always @(posedge `CNN_ENA) begin
  start_time  = $realtime;
  start_ena   = 1;
end

always @(negedge `CNN_ENA) begin
  if (start_ena) begin
    end_time  = $realtime;
    start_ena = 0;
    clkena1   = 1;
  end
end

always @(posedge `CNN_CLK) begin
  if (clkena1) begin
    clk1_time = $realtime;
    clkena1   = 0;
    clkena2   = 1;
  end else if (clkena2) begin
    clk2_time = $realtime;
    clkena2   = 0;
    $display("CNN Cycles = %.0f", $ceil((end_time - start_time)/(clk2_time - clk1_time)));
  end
end
