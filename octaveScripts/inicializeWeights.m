function [wx,wxb,ws,wsb] = inicializeWeights(input_layer,hidden_layer,output_layer)
  
  wx = (rand(hidden_layer,input_layer)*10)-5;
  wxb = (rand(hidden_layer,1))*10-5;
  ws = (rand(output_layer,hidden_layer)*10)-5;
  wsb = (rand(output_layer,1)*10)-5;
  
  return;
endfunction