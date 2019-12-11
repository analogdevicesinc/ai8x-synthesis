// Check default register values.
// Write all registers.
// Make sure only writable bits will change.
int     inp1;
string  fn;
 
initial begin
  //----------------------------------------------------------------
  // Initialize the CNN
  //----------------------------------------------------------------
  #200000;
  fn = {`TARGET_DIR,"/input.mem"};
  inp1 = $fopen(fn, "r");
  if (inp1 == 0) begin
    $display("ERROR : CAN NOT OPEN THE FILE");
  end
  else begin
    $display("Successfully opened input.mem");
    write_cnn(inp1);
    $fclose(inp1);
  end
end
 
initial begin
  #1;
  error_count = 0;
  @(posedge rstn);
  #5000;     // for invalidate done
  -> StartTest;
end

